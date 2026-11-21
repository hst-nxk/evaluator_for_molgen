# What PBCNet 2.0 actually feeds its network, and what that says about ours

Task 6 of the 2026-11-09 meeting: *read PBCNet 2.0 again and establish exactly how its
input is designed, then find the ICML paper whose single input is the 3D complex after
docking and which argues a uni-modal complex beats separate multi-modal branches.*

This note answers the first half from the authors' own code (not from the paper's prose),
and records honestly that the second half could not be closed.

Source: `https://github.com/YuJie-0202/PBCNet2.0`, branch `master`, read 2026-11-21. The
repository is MIT licensed; nothing from it is copied into this project. Line references
are to that commit.

## 1. PBCNet 2.0's input is a 3D protein-ligand complex graph, not molecular descriptors

Built by `Graph2pickle.py`, from a ligand SDF plus a `pocket.pdb` per complex:

* **Nodes** are *atoms*, hydrogens removed, of the ligand AND of the pocket, concatenated
  into one node set with 3D coordinates from each file's conformer
  (`Graph_Information`, the `pos` block). So the protein is not a separate branch: it is
  part of the same graph as the ligand.
* **Edges** (`etype='int'`) are built in four passes:
  1. ligand-ligand, every atom pair closer than **5 Å**;
  2. pocket-pocket covalent bonds, from RDKit;
  3. ligand-pocket, every pair closer than **5 Å** - the interaction edges;
  4. pocket-pocket by distance under **3 Å**, but only from pocket atoms that pass 3,
     i.e. a second shell only around atoms the ligand can reach.
* **Node features**: ligand-style atom features for both sides (`lig_atom_featurizer`),
  plus the residue identity and a five-way residue property vector for pocket atoms
  (`res_pocket`, `prop`: aliphatic / aromatic / polar-neutral / acidic / basic), plus
  `type`, a 0/1 flag marking ligand atoms.
* **Edge features**: a bond type scalar, clipped to five classes in the model.

So the geometry *is* the protein representation: distances and direction vectors between
ligand and pocket atoms. There is no global protein embedding anywhere in the model.

## 2. The encoder is equivariant, and the protein is masked at readout

`model_code/models/readout.py`:

* The encoder is **TensorNet** (Cartesian tensor representations, `O(3)`,
  `hidden_channels=128`, 5 layers, 32 radial basis functions, cutoff 0-5 Å). Before it
  runs, the graph is given edge-wise distances and vectors computed from `pos`
  (`src_norm_dst` and `fn.v_sub_u`).
* After encoding, atom embeddings are multiplied by the ligand flag -
  `emb * mask  # mask the protein atoms` - and only then summed per graph
  (`dgl.readout_nodes`). The protein therefore shapes the ligand's atom embeddings through
  message passing but contributes no term of its own to the pooled vector.

## 3. The difference is the WHOLE representation, and both directions are computed

```python
emb1, att1 = self._readout(g1, bb)      # complex of ligand 1
emb2, att2 = self._readout(g2, bb)      # complex of ligand 2
emb      = self.norm(emb1 - emb2)
out_put  = self.FNN(emb)
emb_neg  = self.norm(emb2 - emb1)
out_put_neg = self.FNN(emb_neg)
return out_put, out_put_neg
```

Two things to notice, because both differ from what this project tried:

1. **The subtraction is the only thing the head ever sees.** `emb1 - emb2` goes through a
   LayerNorm and a four-layer FFN (128 -> 256 -> 128 -> 64 -> 1). There is no
   concatenation of the two embeddings beside the difference, and no separate protein
   vector to concatenate.
2. **Antisymmetry is imposed by the LOSS, not by the architecture.** The head is a
   nonlinear FFN after a LayerNorm, so `f(A,B) = -f(B,A)` does not hold by construction.
   `model_code/train.py` trains both outputs against opposite labels:

   ```python
   label_neg = torch.neg(label)
   loss = loss_func(logits, label) + loss_func(logits_neg, label_neg)
   ```

## 4. What this means for the numbers already measured here

| the meeting's item | what was measured here | what PBCNet 2.0 actually does |
|---|---|---|
| task 3: "add the A - B difference feature" | an EXTRA token `enc(A) - enc(B)` appended to a concatenation: +0.0001 pairwise macro, and +0.0010 with the product term, both under the 0.0021 noise floor | the difference REPLACES the concatenation - the head sees nothing else |
| task 4: "force symmetry through the loss" | `antisymmetry_penalty=1.0`: 0.5541 +/- 0.0002, +0.0041 over the control, t = 10.48 - the largest of the three symmetry routes | the same mechanism: a loss term on the reversed direction, both directions through one head |
| task 5: "an architecture that satisfies symmetry by construction" | `twin` and `antisymmetric` heads: asymmetry exactly 0.0000, cost-free on Morgan (+0.0011), -0.0020 on MoLFormer | NOT what PBCNet 2.0 does; its head is not antisymmetric by construction |
| the protein branch | a global ESM2 vector concatenated to the ligand features: removing it changes nothing (+0.0005 on SUB100, signs mixed; +0.0030 better with the column shuffled) | no global protein vector exists; the protein enters as 3D geometry within 5 Å of the ligand |

Two consequences for how this work may be written up:

* **It is still not true to write "we tried PBCNet 2.0's design".** Their input is a docked
  3D complex per ligand; this project's dataset has no poses, so their design is not
  reachable without a docking stage. What was tried is the difference/product FEATURE as
  the minutes described it, and that is all that may be claimed.
* **The protein-branch null result here is consistent with their design rather than with
  the minutes' reading of it.** A global sequence embedding of the whole protein carries
  little about one pocket; PBCNet 2.0 never uses one. "Find another protein
  representation" (2026-11-02 meeting) is therefore better read as "represent the pocket,
  not the protein", which needs structures.
* Their task-4 mechanism matching this project's largest symmetry effect is independent
  support for the `antisymmetry_penalty` route over the architectural one.

## 5. The ICML paper of task 6 was NOT found

Searched 2026-11-21 for an ICML paper whose only input is the post-docking 3D complex and
which argues a uni-modal complex beats separate multi-modal branches. **No paper matching
that claim was located**, so the minutes' statement remains unverified and must not be
cited. What came closest, with the way each one differs:

* **TerraBind** (ICML 2026 poster, `arXiv:2602.07735`) - coarse structural representation
  (protein C-beta atoms, ligand heavy atoms), but it is explicitly MULTIMODAL: pretrained
  molecular encoders plus ESM-2 protein embeddings. It supports "a simplified 3D
  representation is enough", not "uni-modal beats multi-modal".
* **MIRAGE** (`arXiv:2609.14491`) - runs uni-modal controls. A protein-only model nearly
  matches the full model under random splits and falls sharply under family-disjoint
  splits, while a ligand-only model degrades less. This is directly relevant to this
  project: SUB100 splits by protein cluster, and there the protein branch adds nothing -
  the same direction MIRAGE reports.
* **EMBER** (`arXiv:2610.27334`) - the nearest architectural ablation (one shared pass over
  a co-folded complex against encoding each molecule separately), but on TCR-pMHC binding,
  not relative affinity.

To close this item, the meeting needs to supply the title, the authors or the arXiv id.
Until then the paper is listed here as unverified rather than cited.
