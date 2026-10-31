"""IGNITE: phạt chuẩn gradient đầu ra của surrogate để mô hình bớt "sắc".

Nguồn: Dao MC, Nguyen PL, Truong TN, Hoang TN. *Incorporating Surrogate Gradient Norm to
Improve Offline Optimization Techniques*. NeurIPS 2024, DOI 10.52202/079017-0258,
arXiv:2503.04242.

Ý tưởng của bài báo: một surrogate train trên dữ liệu offline thường không đáng tin ở vùng
nằm ngoài dữ liệu đã thấy. Bài báo đo mức "sắc" của surrogate bằng

    R_D(omega) = max_{||delta|| <= rho} | E_x[g(x; omega + delta)] - E_x[g(x; omega)] |

và xấp xỉ nó bằng ``rho * ||grad_omega h(omega)||`` với ``h(omega) = E_x[g(x; omega)]``
(công thức 10, 18). Vì vậy thay vì chỉ tối thiểu hóa mất mát, bài báo giải bài toán có
ràng buộc ``rho * ||grad_omega h(omega)|| <= epsilon`` (công thức 11) bằng nhân tử Lagrange
``lambda`` cập nhật theo dual ascent (công thức 12-14).

Điểm thực thi quan trọng: gradient của chính số hạng phạt sẽ cần Hessian, nên bài báo xấp
xỉ nó bằng sai phân hữu hạn quanh điểm bị làm nhiễu ``omega + r * g2 / ||g2||`` (công thức
83-85, Algorithm 1). Nhờ vậy mỗi bước chỉ cần gradient bậc nhất, nhưng phải forward thêm
một lượt.

Gradient tổng của một bước, đúng theo Algorithm 1 của bài báo:

    g1 = grad_omega (mất mát trung bình trên batch)
    g2 = grad_omega (đầu ra trung bình trên batch)      # = grad_omega h
    omega_hat = omega + r * g2 / ||g2||
    g3 = grad_omega (đầu ra trung bình trên batch, tính tại omega_hat)
    g  = g1 + lambda * (rho / r) * (g3 - g2)
    lambda <- lambda + eta_lambda * (rho * ||g2|| - epsilon)

Hai chỗ lệch với bài báo, có chủ ý, đã ghi trong ``stage3.train.sharpness.note`` của
config: bài báo cập nhật tham số bằng gradient descent thuần (ở đây dùng AdamW cho khớp
các lần train trước), và giả thiết 1 của bài báo cần đầu ra bị chặn trong [0, 1] trong khi
nhãn Δp của dự án không bị chặn — nên phần lý thuyết không áp dụng trực tiếp và hiệu quả
phải đo bằng ablation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

SETTING_ENABLED = "enabled"


def is_enabled(settings: Mapping[str, Any], override: str | None = None) -> bool:
    """Có bật số hạng phạt hay không: ``override`` ('on'/'off') thắng config."""
    if override == "on":
        return True
    if override == "off":
        return False
    return bool(settings[SETTING_ENABLED])


class SharpnessPenalty:
    """Một bước IGNITE: tính gradient tổng rồi ghi vào ``.grad`` của từng tham số.

    Nhận: mục ``stage3.train.sharpness`` của config.
    Giữ trạng thái: ``lambda_value`` (nhân tử Lagrange, cập nhật sau mỗi batch).
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.rho = float(settings["rho"])
        self.radius = float(settings["finite_difference_radius"])
        self.epsilon = float(settings["epsilon"])
        self.lambda_learning_rate = float(settings["lambda_learning_rate"])
        self.lambda_value = float(settings["lambda_init"])
        if self.radius <= 0.0:
            raise ValueError("finite_difference_radius phải > 0: nó là mẫu số của sai phân")
        if self.rho <= 0.0:
            raise ValueError("rho phải > 0")
        self.last_gradient_norm = 0.0

    def apply(self, loss: Any, prediction: Any, parameters: Sequence[Any],
              recompute_mean: Callable[[], Any], share: float = 1.0,
              accumulate: bool = False) -> None:
        """Ghi gradient tổng vào ``.grad``, rồi cập nhật ``lambda``.

        Nhận: mất mát của batch; đầu ra của batch (chưa lấy trung bình); danh sách tham số
        cần gradient; hàm chạy lại forward trên ĐÚNG batch đó và trả về đầu ra trung bình
        (cần vì sai phân hữu hạn phải đo ở tham số đã bị làm nhiễu).

        ``share`` nhân vào gradient tổng, và ``accumulate`` thì CỘNG vào ``.grad`` sẵn
        có thay vì ghi đè. Hai tham số này để cộng dồn gradient qua nhiều batch con:
        ghi đè sẽ xóa mất gradient của những batch trước mà không báo gì. Phải nhân
        trọng số vào gradient TỔNG chứ không phải vào ``loss``, vì số hạng phạt tính từ
        ``prediction.mean()`` nên không chịu ảnh hưởng của việc chia nhỏ mất mát.
        """
        import torch

        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("không có tham số nào cần gradient")
        loss_grad = torch.autograd.grad(loss, trainable, retain_graph=True)
        output_grad = torch.autograd.grad(prediction.mean(), trainable)
        norm = torch.sqrt(sum((part * part).sum() for part in output_grad))
        self.last_gradient_norm = float(norm.detach().to("cpu"))

        if self.last_gradient_norm > 0.0:
            scale = self.radius / self.last_gradient_norm
            with torch.no_grad():
                for parameter, part in zip(trainable, output_grad):
                    parameter.add_(part, alpha=scale)
            try:
                shifted_grad = torch.autograd.grad(recompute_mean(), trainable)
            finally:
                # Phải trả tham số về chỗ cũ kể cả khi lượt forward thêm bị lỗi, nếu không
                # mô hình bị bỏ lại ở điểm đã làm nhiễu.
                with torch.no_grad():
                    for parameter, part in zip(trainable, output_grad):
                        parameter.sub_(part, alpha=scale)
            weight = self.lambda_value * self.rho / self.radius
            total = [first + weight * (third - second)
                     for first, second, third in zip(loss_grad, output_grad, shifted_grad)]
        else:
            # ||g2|| = 0: số hạng phạt không có hướng nào để đi, chỉ còn gradient mất mát.
            total = list(loss_grad)

        for parameter, part in zip(trainable, total):
            scaled = part if share == 1.0 else part * share
            if accumulate and parameter.grad is not None:
                parameter.grad = parameter.grad + scaled
            else:
                parameter.grad = scaled

        # Dual ascent. Bài báo không nói lambda có bị chặn dưới hay không; chặn tại 0 vì
        # dạng Lagrange ở công thức 12 yêu cầu lambda > 0.
        violation = self.rho * self.last_gradient_norm - self.epsilon
        self.lambda_value = max(0.0, self.lambda_value
                                + self.lambda_learning_rate * violation)

    def state(self) -> dict[str, float]:
        """Trạng thái để ghi vào manifest của lần chạy."""
        return {"lambda": self.lambda_value,
                "output_gradient_norm": self.last_gradient_norm,
                "sharpness_estimate": self.rho * self.last_gradient_norm}
