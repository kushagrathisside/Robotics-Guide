# M2 — Matrices and Linear Maps

> **Mathematics Primer · Chapter 2 of 8** · Previous: [M1 — Vectors and Geometry](M1-vectors-and-geometry.md) · [Primer home](README.md) · Next: [M3 — Eigenvalues, SVD and Least Squares](M3-eigen-svd-least-squares.md)

## Why this matters in robotics

Every computation in manipulator kinematics is a matrix computation. A rotation matrix ${}^{A}_{B}R$ converts coordinates from one frame to another; a $4\times4$ homogeneous transform ${}^{A}_{B}T$ adds a translation; a chain of such transforms multiplied together gives the forward kinematics of an arm; and the Jacobian matrix $J(\boldsymbol{\theta})$ maps joint velocities to tool velocities. Whether a Jacobian is invertible, what its rank is, and which joint motions lie in its null space decide whether a robot is at a singularity and whether it has redundant degrees of freedom. This chapter develops matrices not as tables of numbers but as **linear maps**, because that is the point of view that makes robotics formulas self-evident.

## Learning objectives

After studying this chapter you should be able to:

1. read the columns of a matrix as the images of the basis vectors, and build a matrix from that description;
2. multiply matrices, explain why the order matters, and relate products to composition of maps;
3. use the transpose, the inverse and the determinant, and interpret the determinant as a signed volume factor;
4. compute rank and null space, and state the rank–nullity theorem;
5. recognise orthogonal matrices and the special orthogonal group $SO(3)$;
6. change coordinates between frames and transform a linear operator by similarity.

## Contents

1. [Matrices as linear maps](#1-matrices-as-linear-maps)
2. [Matrix products and composition](#2-matrix-products-and-composition)
3. [Transpose, symmetric and skew-symmetric matrices](#3-transpose-symmetric-and-skew-symmetric-matrices)
4. [Inverse and linear systems](#4-inverse-and-linear-systems)
5. [Determinant: volume and orientation](#5-determinant-volume-and-orientation)
6. [Rank, column space and null space](#6-rank-column-space-and-null-space)
7. [Orthogonal matrices and SO(3)](#7-orthogonal-matrices-and-so3)
8. [Change of basis and homogeneous coordinates](#8-change-of-basis-and-homogeneous-coordinates)
9. [Worked examples](#9-worked-examples)
10. [Exercises](#10-exercises)
11. [Further reading](#11-further-reading)

---

## 1. Matrices as linear maps

A function $f:\mathbb{R}^n \to \mathbb{R}^m$ is **linear** if $f(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha f(\mathbf{x}) + \beta f(\mathbf{y})$ for all vectors $\mathbf{x}, \mathbf{y}$ and scalars $\alpha, \beta$. Linearity means that $f$ is completely determined by what it does to the basis vectors $\mathbf{e}_1, \dots, \mathbf{e}_n$: since $\mathbf{x} = \sum_j x_j\mathbf{e}_j$, we have $f(\mathbf{x}) = \sum_j x_j f(\mathbf{e}_j)$. Collect the images $f(\mathbf{e}_j)$ as the columns of an $m\times n$ array and you have the **matrix** of $f$:

$$
A = \begin{bmatrix} | & | & & | \\ f(\mathbf{e}_1) & f(\mathbf{e}_2) & \cdots & f(\mathbf{e}_n) \\ | & | & & | \end{bmatrix}, \qquad f(\mathbf{x}) = A\mathbf{x} = \sum_{j=1}^{n} x_j\,(\text{column } j \text{ of } A).
$$

> **Key idea.** *The $j$-th column of a matrix is where the $j$-th basis vector goes.*

This single idea explains the rotation matrix of Craig Chapter 2. The rotation by angle $\theta$ about the $z$-axis sends $\hat{\mathbf{x}}$ to $(\cos\theta, \sin\theta, 0)$, sends $\hat{\mathbf{y}}$ to $(-\sin\theta, \cos\theta, 0)$ and leaves $\hat{\mathbf{z}}$ alone, so

$$
R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}, \quad
R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}.
$$

Likewise, Craig's rotation matrix ${}^{A}_{B}R = \begin{bmatrix} {}^{A}\hat{X}_B & {}^{A}\hat{Y}_B & {}^{A}\hat{Z}_B \end{bmatrix}$ has as its columns the unit axes of frame $\lbrace B\rbrace$ written in frame $\lbrace A\rbrace$.

Two other useful readings of $A\mathbf{x}$: the $i$-th entry of $A\mathbf{x}$ is the dot product of row $i$ of $A$ with $\mathbf{x}$ (the "row picture"), and $A\mathbf{x}$ is a linear combination of the columns of $A$ (the "column picture"). Both are used constantly.

## 2. Matrix products and composition

If $B$ maps $\mathbb{R}^p \to \mathbb{R}^n$ and $A$ maps $\mathbb{R}^n \to \mathbb{R}^m$, then "first $B$, then $A$" is the linear map $\mathbf{x} \mapsto A(B\mathbf{x})$, whose matrix is the **product** $AB$ with entries

$$
(AB)_{ik} = \sum_{j=1}^{n} A_{ij}B_{jk}.
$$

The product is defined only when the inner dimensions agree, and it is **associative**, $(AB)C = A(BC)$, and **distributive**, but in general **not commutative**: $AB \neq BA$. For rotations, non-commutativity is a physical fact. Pick up a book, turn it $90^\circ$ about a vertical axis and then $90^\circ$ about a horizontal axis; repeat in the opposite order; the book ends in different orientations (Example M2.2).

In robotics, products appear as **chains of transforms**:

$$
{}^{0}_{N}T = {}^{0}_{1}T\;{}^{1}_{2}T\;\cdots\;{}^{N-1}_{\phantom{N}N}T .
$$

The subscripts and superscripts "cancel" like a chain, which is a useful check on whether a product is written in the correct order (Craig Ch. 2).

## 3. Transpose, symmetric and skew-symmetric matrices

The **transpose** $A^{T}$ swaps rows and columns, $(A^{T})_{ij} = A_{ji}$. Its key properties are $(A^{T})^{T} = A$, $(A + B)^{T} = A^{T} + B^{T}$ and, note the reversal,

$$
(AB)^{T} = B^{T}A^{T}.
$$

The dot product is a matrix product, $\mathbf{x}\cdot\mathbf{y} = \mathbf{x}^{T}\mathbf{y}$, and so $(A\mathbf{x})\cdot\mathbf{y} = \mathbf{x}\cdot(A^{T}\mathbf{y})$. This identity is behind the duality between velocity and force in Craig Chapter 5: if $J$ maps joint rates to tool velocity, $\mathbf{v} = J\dot{\boldsymbol{\theta}}$, then $J^{T}$ maps tool force to joint torque, $\boldsymbol{\tau} = J^{T}\mathbf{F}$, because power is the same in both descriptions: $\mathbf{F}^{T}\mathbf{v} = \mathbf{F}^{T}J\dot{\boldsymbol{\theta}} = (J^{T}\mathbf{F})^{T}\dot{\boldsymbol{\theta}}$.

A square matrix is **symmetric** if $A^{T} = A$ (inertia tensors, mass matrices, covariance matrices) and **skew-symmetric** if $A^{T} = -A$ (the cross-product matrix $[\boldsymbol{\omega}]_\times$ of [M1](M1-vectors-and-geometry.md)). Every square matrix splits uniquely as $A = \tfrac12(A + A^{T}) + \tfrac12(A - A^{T})$, symmetric plus skew-symmetric.

## 4. Inverse and linear systems

A square matrix $A$ is **invertible** (non-singular) if there is a matrix $A^{-1}$ with $AA^{-1} = A^{-1}A = I$. Geometrically, the map can be undone. The following statements are equivalent for an $n\times n$ matrix:

* $A$ is invertible;
* $\det A \neq 0$;
* $\operatorname{rank} A = n$;
* $A\mathbf{x} = \mathbf{0}$ only for $\mathbf{x} = \mathbf{0}$;
* $A\mathbf{x} = \mathbf{b}$ has exactly one solution for every $\mathbf{b}$.

Useful rules: $(AB)^{-1} = B^{-1}A^{-1}$ and $(A^{T})^{-1} = (A^{-1})^{T}$. For a $2\times2$ matrix,

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}.
$$

In general $A^{-1} = \operatorname{adj}(A)/\det A$, where the adjugate is the transposed matrix of cofactors. This formula is valuable for small symbolic matrices (for example, inverting a $2\times2$ Jacobian by hand in Craig Ch. 5), but it is a poor numerical method. **In computation, solve $A\mathbf{x} = \mathbf{b}$ by Gaussian elimination or an LU, QR or Cholesky factorisation (for example `numpy.linalg.solve`) rather than forming $A^{-1}$.** It is faster and more accurate.

## 5. Determinant: volume and orientation

The **determinant** of a square matrix is the factor by which the linear map scales signed volume (area in two dimensions). The unit square spanned by $\mathbf{e}_1, \mathbf{e}_2$ is sent to the parallelogram spanned by the columns of $A$, whose signed area is $\det A$. If $\det A < 0$ the map reverses orientation: a right-handed triple becomes left-handed, like a mirror. If $\det A = 0$ the map squashes space onto a lower-dimensional subspace and cannot be inverted.

For $3\times 3$ matrices the determinant equals the scalar triple product of the columns (or rows), $\det\begin{bmatrix}\mathbf{a} & \mathbf{b} & \mathbf{c}\end{bmatrix} = \mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})$. It can be computed by cofactor expansion along any row or column.

| Property | Statement |
|---|---|
| Products | $\det(AB) = \det A\,\det B$ |
| Transpose | $\det A^{T} = \det A$ |
| Inverse | $\det A^{-1} = 1/\det A$ |
| Scaling | $\det(cA) = c^{n}\det A$ for $n\times n$ $A$ |
| Row operations | swapping two rows changes the sign; adding a multiple of one row to another changes nothing |
| Triangular matrices | $\det$ is the product of the diagonal entries |

**Where this is used.** A square Jacobian is singular exactly where $\det J(\boldsymbol{\theta}) = 0$. For the planar two-link arm, $\det J = l_1 l_2\sin\theta_2$, which vanishes when the arm is fully stretched or folded (Craig Ch. 5; Lynch & Park §5.3). The value $|\det J|$ is the area scale factor from joint-velocity space to Cartesian-velocity space, which is the manipulability measure previewed in [M3](M3-eigen-svd-least-squares.md).

## 6. Rank, column space and null space

For an $m\times n$ matrix $A$:

* the **column space** (range) $\mathcal{R}(A) = \lbrace A\mathbf{x} \rbrace$ is the set of all outputs, a subspace of $\mathbb{R}^m$;
* the **null space** (kernel) $\mathcal{N}(A) = \lbrace \mathbf{x} : A\mathbf{x} = \mathbf{0} \rbrace$ is the set of inputs that produce no output, a subspace of $\mathbb{R}^n$;
* the **rank** is $\dim\mathcal{R}(A)$, the number of linearly independent columns, which always equals the number of linearly independent rows.

**Rank–nullity theorem.** $\operatorname{rank}A + \dim\mathcal{N}(A) = n$, the number of columns.

To find the null space, row-reduce $A$ to echelon form, identify the free variables, and solve for the pivot variables in terms of them; each free variable contributes one basis vector.

**Where this is used.** For a robot with $n$ joints and an $m$-dimensional task, $J$ is $m\times n$.

* If $\operatorname{rank}J < m$, some task-space directions cannot be produced: the robot is at a **singularity**.
* If $n > m$ (a **redundant** manipulator, such as a 7-joint arm doing a 6-D task), $\mathcal{N}(J)$ has dimension at least $n - m$. Joint velocities $\dot{\boldsymbol{\theta}} \in \mathcal{N}(J)$ move the arm internally without moving the tool: the **self-motion** used to avoid obstacles or joint limits (Siciliano Ch. 3; Lynch & Park Ch. 6).
* The joint torques $\boldsymbol{\tau} = J^{T}\mathbf{F}$ can resist only those wrenches not in $\mathcal{N}(J^{T})$; at a singularity some wrenches are carried by the structure with no torque at all.

## 7. Orthogonal matrices and SO(3)

A square matrix $Q$ is **orthogonal** if its columns are orthonormal (mutually perpendicular unit vectors). Equivalently,

$$
Q^{T}Q = QQ^{T} = I, \qquad\text{so}\qquad Q^{-1} = Q^{T}.
$$

Orthogonal matrices preserve dot products, $(Q\mathbf{x})\cdot(Q\mathbf{y}) = \mathbf{x}^{T}Q^{T}Q\mathbf{y} = \mathbf{x}\cdot\mathbf{y}$, and therefore preserve lengths and angles: they are rigid motions that fix the origin. Taking determinants of $Q^{T}Q = I$ gives $(\det Q)^2 = 1$, so $\det Q = \pm1$. Those with $\det Q = -1$ include a reflection. The **rotation matrices** are exactly the orthogonal matrices with determinant $+1$; they form the **special orthogonal group**

$$
SO(3) = \lbrace R \in \mathbb{R}^{3\times3} : R^{T}R = I,\ \det R = +1 \rbrace .
$$

The product of two rotations is a rotation and the inverse of a rotation is its transpose, which is why Craig writes ${}^{B}_{A}R = {}^{A}_{B}R^{-1} = {}^{A}_{B}R^{T}$. The nine entries of $R$ satisfy six independent constraints (three unit-length and three orthogonality conditions), leaving **three** degrees of freedom; this is why orientation can be described by three angles (Euler or roll–pitch–yaw angles, Craig Ch. 2).

## 8. Change of basis and homogeneous coordinates

### Coordinates of a vector

Suppose frames $\lbrace A\rbrace$ and $\lbrace B\rbrace$ share an origin. A vector has coordinates ${}^{B}\mathbf{P}$ in $\lbrace B\rbrace$, meaning $\mathbf{P} = {}^{B}p_x\hat{X}_B + {}^{B}p_y\hat{Y}_B + {}^{B}p_z\hat{Z}_B$. Writing each axis of $\lbrace B\rbrace$ in $\lbrace A\rbrace$ coordinates and using the column picture of Section 1:

$$
{}^{A}\mathbf{P} = {}^{A}_{B}R\;{}^{B}\mathbf{P}.
$$

Note how the leading subscript $B$ of the matrix "cancels" against the leading superscript $B$ of the vector.

### Coordinates of a linear operator

A linear operation, such as "scale by 2 along the long axis of a part", has a matrix $D_B$ when written in $\lbrace B\rbrace$ coordinates. To apply it to a vector given in $\lbrace A\rbrace$ coordinates, convert to $\lbrace B\rbrace$, apply $D_B$, and convert back:

$$
D_A = {}^{A}_{B}R\; D_B\; {}^{A}_{B}R^{T}.
$$

More generally, if $P$ is any invertible change-of-basis matrix, $D_A = P D_B P^{-1}$; matrices related in this way are called **similar** and share their determinant, trace and eigenvalues. The same formula transforms inertia tensors between frames, $I_A = R\,I_B\,R^{T}$ (Craig Ch. 6; [M8](M8-lagrangian-mechanics-primer.md)), and skew matrices, $R[\boldsymbol{\omega}]_\times R^{T} = [R\boldsymbol{\omega}]_\times$ (M1).

### Homogeneous transforms

When the origins differ by ${}^{A}\mathbf{P}_{B\,\text{org}}$, the coordinate change becomes affine, ${}^{A}\mathbf{P} = {}^{A}_{B}R\,{}^{B}\mathbf{P} + {}^{A}\mathbf{P}_{B\,\text{org}}$. Appending a $1$ to every point turns this into a single matrix product with the $4\times4$ **homogeneous transform**

$$
{}^{A}_{B}T = \begin{bmatrix} {}^{A}_{B}R & {}^{A}\mathbf{P}_{B\,\text{org}} \\ \mathbf{0}^{T} & 1 \end{bmatrix}, \qquad
\begin{bmatrix} {}^{A}\mathbf{P} \\ 1 \end{bmatrix} = {}^{A}_{B}T\begin{bmatrix} {}^{B}\mathbf{P} \\ 1 \end{bmatrix}, \qquad
{}^{A}_{B}T^{-1} = \begin{bmatrix} {}^{A}_{B}R^{T} & -{}^{A}_{B}R^{T}\,{}^{A}\mathbf{P}_{B\,\text{org}} \\ \mathbf{0}^{T} & 1 \end{bmatrix}.
$$

The inverse formula exploits $R^{-1} = R^{T}$ and should always be preferred to a general $4\times4$ inversion. Homogeneous transforms are the central object of Craig Ch. 2–3 and Lynch & Park Ch. 3–4 (where the set of such matrices is called $SE(3)$).

---

## 9. Worked examples

### Example M2.1 — Building and applying a rotation

*Write $R_z(30^\circ)$ and rotate the point $\mathbf{p} = (2, 1, 0)^{T}$.*

**Solution.** With $\cos30^\circ = 0.8660$ and $\sin30^\circ = 0.5$,

$$
R_z(30^\circ)\,\mathbf{p} = \begin{bmatrix} 0.8660 & -0.5 & 0 \\ 0.5 & 0.8660 & 0 \\ 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.7321 - 0.5 \\ 1 + 0.8660 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.2321 \\ 1.8660 \\ 0 \end{bmatrix}.
$$

Check: the length is preserved, $\sqrt{1.2321^2 + 1.8660^2} = 2.2361 = \sqrt{5} = \lVert\mathbf{p}\rVert$.

### Example M2.2 — Rotations do not commute

*Compute $R_x(90^\circ)R_z(90^\circ)$ and $R_z(90^\circ)R_x(90^\circ)$, and apply each to $\hat{\mathbf{x}}$.*

**Solution.** With $R_x(90^\circ) = \begin{bmatrix} 1&0&0\\0&0&-1\\0&1&0 \end{bmatrix}$ and $R_z(90^\circ) = \begin{bmatrix} 0&-1&0\\1&0&0\\0&0&1 \end{bmatrix}$:

$$
R_x R_z = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix}, \qquad R_z R_x = \begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}.
$$

The first column of each is the image of $\hat{\mathbf{x}}$: $R_xR_z\hat{\mathbf{x}} = \hat{\mathbf{z}}$ (first $z$ turns $\hat{\mathbf{x}}$ into $\hat{\mathbf{y}}$, then $x$ turns $\hat{\mathbf{y}}$ into $\hat{\mathbf{z}}$), whereas $R_zR_x\hat{\mathbf{x}} = \hat{\mathbf{y}}$ (the $x$-rotation leaves $\hat{\mathbf{x}}$ alone, then $z$ turns it into $\hat{\mathbf{y}}$). Both products are valid rotations, but different ones.

### Example M2.3 — Determinant, inverse and a linear system

*For $A = \begin{bmatrix} 2&1&0\\1&3&1\\0&1&2 \end{bmatrix}$ and $\mathbf{b} = (1, 2, 3)^{T}$, find $\det A$, $A^{-1}$ and the solution of $A\mathbf{x} = \mathbf{b}$.*

**Solution.** Expanding along the first row,

$$
\det A = 2\,(3\cdot2 - 1\cdot1) - 1\,(1\cdot2 - 1\cdot0) + 0 = 10 - 2 = 8 .
$$

The cofactor matrix is symmetric here (because $A$ is), and

$$
A^{-1} = \frac{1}{8}\begin{bmatrix} 5 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 5 \end{bmatrix}, \qquad
\mathbf{x} = A^{-1}\mathbf{b} = \frac{1}{8}\begin{bmatrix} 5 - 4 + 3 \\ -2 + 8 - 6 \\ 1 - 4 + 15 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0 \\ 1.5 \end{bmatrix}.
$$

Check: $A\mathbf{x} = (1 + 0 + 0,\ 0.5 + 0 + 1.5,\ 0 + 0 + 3) = (1, 2, 3)$. ✓

### Example M2.4 — Rank and null space of a wide matrix

*A three-joint mechanism has a $2\times3$ velocity map $J = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix}$. Find its rank and null space, and interpret.*

**Solution.** The two rows are not multiples of each other, so $\operatorname{rank}J = 2$: every planar task velocity can be produced. By rank–nullity, $\dim\mathcal{N}(J) = 3 - 2 = 1$. Let $\dot\theta_3 = t$ be free. The second row gives $\dot\theta_2 + 2t = 0$, so $\dot\theta_2 = -2t$; the first gives $\dot\theta_1 - 2t + t = 0$, so $\dot\theta_1 = t$. Thus

$$
\mathcal{N}(J) = \operatorname{span}\lbrace (1, -2, 1)^{T} \rbrace .
$$

Moving the joints at rates proportional to $(1, -2, 1)$ produces zero task velocity: this is a self-motion of the redundant mechanism.

### Example M2.5 — Changing coordinates of a vector and an operator

*Frame $\lbrace B\rbrace$ is rotated $90^\circ$ about $\hat{Z}_A$ relative to $\lbrace A\rbrace$, with the same origin. (a) Express ${}^{B}\mathbf{P} = (1, 0, 0)^{T}$ in $\lbrace A\rbrace$. (b) The operator "stretch by a factor of 2 along $\hat{X}_B$" is $D_B = \operatorname{diag}(2, 1, 1)$. Find $D_A$.*

**Solution.** (a) ${}^{A}_{B}R = R_z(90^\circ)$, whose first column is $(0,1,0)^{T}$, so ${}^{A}\mathbf{P} = (0, 1, 0)^{T}$: the $x$-axis of $\lbrace B\rbrace$ points along the $y$-axis of $\lbrace A\rbrace$.

(b) $D_A = R_z(90^\circ)\,\operatorname{diag}(2,1,1)\,R_z(90^\circ)^{T} = \operatorname{diag}(1, 2, 1)$. The stretch is along $\hat{Y}_A$, which is the same physical direction as $\hat{X}_B$, as expected. Note that $D_A$ and $D_B$ have the same trace ($4$) and determinant ($2$).

### Example M2.6 — Homogeneous transform and its inverse

*Frame $\lbrace B\rbrace$ is rotated $90^\circ$ about $\hat{Z}_A$ and its origin is at ${}^{A}\mathbf{P}_{B\,\text{org}} = (1, 2, 0)^{T}$. Map ${}^{B}\mathbf{P} = (1, 0, 0)^{T}$ into $\lbrace A\rbrace$ and write ${}^{B}_{A}T$.*

**Solution.** ${}^{A}\mathbf{P} = R_z(90^\circ)(1,0,0)^{T} + (1,2,0)^{T} = (0,1,0)^{T} + (1,2,0)^{T} = (1, 3, 0)^{T}$. For the inverse, $R^{T} = \begin{bmatrix} 0&1&0\\-1&0&0\\0&0&1\end{bmatrix}$ and $-R^{T}(1,2,0)^{T} = -(2, -1, 0)^{T} = (-2, 1, 0)^{T}$, so

$$
{}^{B}_{A}T = \begin{bmatrix} 0 & 1 & 0 & -2 \\ -1 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}.
$$

Check: ${}^{B}_{A}T\,(1, 3, 0, 1)^{T} = (3 - 2,\ -1 + 1,\ 0,\ 1)^{T} = (1, 0, 0, 1)^{T}$. ✓

---

## 10. Exercises

**E2.1** Let $A = \begin{bmatrix} 1&2\\0&1 \end{bmatrix}$ and $B = \begin{bmatrix} 1&0\\3&1 \end{bmatrix}$. Compute $AB$ and $BA$.

<details><summary>Answer</summary>

$AB = \begin{bmatrix} 7&2\\3&1 \end{bmatrix}$ and $BA = \begin{bmatrix} 1&2\\3&7 \end{bmatrix}$. They differ, so $A$ and $B$ do not commute.

</details>

**E2.2** Show that $R_z(\theta)$ is orthogonal and has determinant $+1$ for every $\theta$.

<details><summary>Answer</summary>

The columns $(\cos\theta, \sin\theta, 0)$, $(-\sin\theta, \cos\theta, 0)$ and $(0,0,1)$ each have unit length ($\cos^2\theta + \sin^2\theta = 1$) and their pairwise dot products are $-\cos\theta\sin\theta + \sin\theta\cos\theta = 0$, $0$ and $0$. Hence $R_z^{T}R_z = I$. Expanding along the third row, $\det R_z = \cos^2\theta + \sin^2\theta = 1$.

</details>

**E2.3** Compute $\det\begin{bmatrix} 1&2\\3&4 \end{bmatrix}$ and describe what the map does to areas and orientation.

<details><summary>Answer</summary>

$\det = 1\cdot4 - 2\cdot3 = -2$. Areas are doubled and orientation is reversed (counter-clockwise ordering of vertices becomes clockwise).

</details>

**E2.4** Find the rank and a basis of the null space of $\begin{bmatrix} 1&2&3\\2&4&6 \end{bmatrix}$.

<details><summary>Answer</summary>

The second row is twice the first, so the rank is $1$ and the null space has dimension $3 - 1 = 2$. With free variables $x_2 = s$ and $x_3 = t$, $x_1 = -2s - 3t$, giving the basis $(-2, 1, 0)^{T}$ and $(-3, 0, 1)^{T}$.

</details>

**E2.5** Invert $\begin{bmatrix} 4&7\\2&6 \end{bmatrix}$.

<details><summary>Answer</summary>

$\det = 24 - 14 = 10$, so the inverse is $\dfrac{1}{10}\begin{bmatrix} 6&-7\\-2&4 \end{bmatrix} = \begin{bmatrix} 0.6&-0.7\\-0.2&0.4 \end{bmatrix}$.

</details>

**E2.6** Prove that if $Q$ is orthogonal then $\lVert Q\mathbf{x}\rVert = \lVert\mathbf{x}\rVert$ for every $\mathbf{x}$.

<details><summary>Answer</summary>

$\lVert Q\mathbf{x}\rVert^2 = (Q\mathbf{x})^{T}(Q\mathbf{x}) = \mathbf{x}^{T}Q^{T}Q\mathbf{x} = \mathbf{x}^{T}\mathbf{x} = \lVert\mathbf{x}\rVert^2$, and both norms are non-negative.

</details>

**E2.7** Given ${}^{A}_{B}R = R_z(90^\circ)$ and ${}^{B}_{C}R = R_x(90^\circ)$, find ${}^{A}_{C}R$ and express ${}^{C}\mathbf{P} = (0, 0, 1)^{T}$ in frame $\lbrace A\rbrace$.

<details><summary>Answer</summary>

${}^{A}_{C}R = {}^{A}_{B}R\,{}^{B}_{C}R = R_z(90^\circ)R_x(90^\circ) = \begin{bmatrix} 0&0&1\\1&0&0\\0&1&0 \end{bmatrix}$ (Example M2.2). Its third column gives ${}^{A}\mathbf{P} = (1, 0, 0)^{T}$.

</details>

**E2.8** Explain, using determinants, why a matrix with $\det A = 0$ cannot have an inverse.

<details><summary>Answer</summary>

If $A^{-1}$ existed, then $1 = \det I = \det(AA^{-1}) = \det A\,\det A^{-1} = 0$, a contradiction. Geometrically, $A$ flattens space onto a lower-dimensional subspace, so distinct inputs share an output and the map cannot be undone.

</details>

---

## 11. Further reading

* G. Strang, *Introduction to Linear Algebra*, Ch. 2 (Solving Linear Equations), Ch. 3 (Vector Spaces and Subspaces), Ch. 5 (Determinants) and Ch. 8 (Linear Transformations).
* E. Kreyszig, *Advanced Engineering Mathematics*, Ch. 7 (Linear Algebra: Matrices, Vectors, Determinants, Linear Systems).
* J. J. Craig, *Introduction to Robotics*, Ch. 2 (rotation matrices, homogeneous transforms, transform equations) and Ch. 5 (Jacobians and singularities).
* K. M. Lynch and F. C. Park, *Modern Robotics*, Ch. 3 ($SO(3)$ and $SE(3)$) and Ch. 5 (velocity kinematics, singularities).
* B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Appendix A (Linear Algebra).
* M. W. Spong, S. Hutchinson and M. Vidyasagar, *Robot Modeling and Control*, Ch. 2 (Rigid Motions) and the linear-algebra appendix.

---

Previous: [M1 — Vectors and Geometry](M1-vectors-and-geometry.md) · [Primer home](README.md) · Next: [M3 — Eigenvalues, SVD and Least Squares](M3-eigen-svd-least-squares.md)
