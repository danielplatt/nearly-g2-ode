# Berger nearly-parallel `G2` ODEs: Codex handoff

Source: `SO(4)-invariant G2 structures on the Berger space`, especially equations (3.8), (4.6), (4.7), (4.9), and Section 5.2.  This note uses the left singular orbit at `t = 0`, whose singular isotropy has weights `(3,-1)`.

The scalar `lambda` is kept in the equations because the ODEs require it.  For the intended use here, treat `lambda` as fixed, not as a shooting parameter.

## 1. Ansatz and variables

Use the invariant `G2` form

```text
phi = (p1 e12 + p2 e34 + p3 e56) wedge dt
    + q1 e135 + q2 e136 + q3 e145 + q4 e146
    + q5 e235 + q6 e236 + q7 e245 + q8 e246.
```

Here

```text
q_i = q_i(t),  i = 1,...,8,
p_j = p_j(t),  j = 1,2,3.
```

Define

```text
alpha1 = q1*q8
alpha2 = q2*q7
alpha3 = q3*q6
alpha4 = q4*q5
A      = alpha1 + alpha2 + alpha3 + alpha4

beta1 = q1*q4*q6*q7
beta2 = q2*q3*q5*q8
```

For compact implementation, define

```text
C1_beta1 = alpha1*(2*alpha1 - A) + 2*beta1
C1_beta2 = alpha1*(2*alpha1 - A) + 2*beta2
C2_beta1 = alpha2*(2*alpha2 - A) + 2*beta1
C2_beta2 = alpha2*(2*alpha2 - A) + 2*beta2
C3_beta1 = alpha3*(2*alpha3 - A) + 2*beta1
C3_beta2 = alpha3*(2*alpha3 - A) + 2*beta2
C4_beta1 = alpha4*(2*alpha4 - A) + 2*beta1
C4_beta2 = alpha4*(2*alpha4 - A) + 2*beta2
```

## 2. Nearly-parallel `G2` ODE system

The nearly-parallel equation is

```text
d phi = lambda * psi.
```

It imposes the algebraic constraints

```text
q2 + q7 = -lambda*p1*p2,
q3 + q6 = -lambda*p1*p3,
q4 + q5 = -lambda*p2*p3.
```

On the left singular branch used below, compute `p_i` from `q_i` by

```text
p1 = -sqrt( -((q2 + q7)*(q3 + q6))/(lambda*(q4 + q5)) )
p2 =  sqrt( -((q2 + q7)*(q4 + q5))/(lambda*(q3 + q6)) )
p3 =  sqrt( -((q3 + q6)*(q4 + q5))/(lambda*(q2 + q7)) )
```

The sign of `p1` is part of the branch choice.  It is the sign used for the Riemannian `mu = -1` branch near `t = 0`.

With these conventions, the ODE system for `q = (q1,...,q8)` is

```text
q1' =  lambda/(2*p1*p2*p3*q8) * C1_beta2
q8' = -lambda/(2*p1*p2*p3*q1) * C1_beta1

q2' =  p3 - lambda/(2*p1*p2*p3*q7) * C2_beta1
q7' = -p3 + lambda/(2*p1*p2*p3*q2) * C2_beta2

q3' =  p2 - lambda/(2*p1*p2*p3*q6) * C3_beta1
q6' = -p2 + lambda/(2*p1*p2*p3*q3) * C3_beta2

q4' = -p1 + lambda/(2*p1*p2*p3*q5) * C4_beta2
q5' =  p1 - lambda/(2*p1*p2*p3*q4) * C4_beta1
```

Do not evaluate this singular `q`-system at `t = 0`, because several denominators vanish there.  Start at a small `eps > 0` using the Taylor data below, or integrate a regularized `y_i` system.

The SU(3) normalization/discriminant condition to monitor is

```text
Delta = sum_{i<j} (alpha_i - alpha_j)^2 - 2*sum_i alpha_i^2 + 4*(beta1 + beta2)
Delta = -4*(p1*p2*p3)^2.
```

## 3. Smooth left-end data for the `mu = -1` family

For the smooth extension over the left singular orbit, write

```text
q1(t) = a    + t^2*y1(t)
q2(t) =        t*y2(t)
q3(t) =        t*y3(t)
q4(t) = c    + t^2*y4(t)
q5(t) = -3*a + t^2*y5(t)
q6(t) =        t*y6(t)
q7(t) =        t*y7(t)
q8(t) = -3*c + t^2*y8(t).
```

For the Riemannian Berger branch assume

```text
a > 0,
c < 0,
D = 3*a - c > 0,
lambda > 0.
```

Set

```text
m = sqrt(-a*c)
r = sqrt(D/lambda)
s = sqrt(-lambda*a*c*D)
u = sqrt(-lambda*a*c/D)
```

For `mu = -1`, the zeroth-order `y_i(0) = b_i` values are

```text
b1 = -(3*lambda^2*a*(5*a - c))/(8*D) - s/(2*c)
b2 = -lambda*m + r
b3 = b2
b4 = -(lambda^2*c*(9*a - 5*c))/(8*D) - s/(2*a) + u
b5 =  (9*lambda^2*a*(5*a - c))/(8*D) + s/(2*c) - u
b6 =  3*lambda*m - r
b7 = b6
b8 =  (3*lambda^2*c*(9*a - 5*c))/(8*D) + s/(2*a)
```

The third parameter is

```text
nu = y3'(0).
```

The remaining first derivatives are fixed by smoothness and the `mu = -1` branch:

```text
y1'(0) = 0
y4'(0) = 0
y5'(0) = 0
y8'(0) = 0

y2'(0) = -nu
y3'(0) =  nu

K = (sqrt(D) + 3*lambda^(3/2)*sqrt(-a*c)) / (sqrt(D) - lambda^(3/2)*sqrt(-a*c))

y6'(0) = -K*nu
y7'(0) =  K*nu
```

Equivalently, for numerical integration from `t = eps`, use

```text
q1(eps) = a    + eps^2*b1 + O(eps^4)
q2(eps) = eps*b2 - eps^2*nu + O(eps^3)
q3(eps) = eps*b2 + eps^2*nu + O(eps^3)
q4(eps) = c    + eps^2*b4 + O(eps^4)
q5(eps) = -3*a + eps^2*b5 + O(eps^4)
q6(eps) = eps*b6 - eps^2*K*nu + O(eps^3)
q7(eps) = eps*b6 + eps^2*K*nu + O(eps^3)
q8(eps) = -3*c + eps^2*b8 + O(eps^4).
```

This is the three-parameter local family usually denoted `eta_{a,c,nu}`.  The parameter `lambda` is an additional scaling/torsion constant; here it is treated as fixed.

A useful check against the homogeneous Berger solution is

```text
a      = sqrt(5)/20
c      = -3*sqrt(5)/100
nu    = sqrt(5)/50
lambda = 6/sqrt(5)
```

## 4. Meaning of `a`, `c`, and `nu`

The parameters `a` and `c` are intrinsic singular-orbit parameters.  They are already visible in the value of the `G2` form at `t = 0`:

```text
q1(0) = a,
q4(0) = c,
q5(0) = -3*a,
q8(0) = -3*c,
q2(0) = q3(0) = q6(0) = q7(0) = 0.
```

They determine the metric induced on the singular orbit `Sigma_-`.  In the basis used in the paper, this metric is diagonal with entries

```text
-25*lambda*a*c/(3*a - c),
 a*sqrt((3*a - c)/(lambda*abs(a*c))),
-c*sqrt((3*a - c)/(lambda*abs(a*c))),
 a*sqrt((3*a - c)/(lambda*abs(a*c))),
-c*sqrt((3*a - c)/(lambda*abs(a*c))).
```

Thus the induced metric is Riemannian exactly on the branch `a > 0`, `c < 0`.  The same data can be summarized by saying that `a` and `-c` are the volume-density parameters of the two distinguished singular-orbit distributions: in the paper's notation,

```text
sqrt(det(g_{T^{3,-1}} restricted to Sigma_1)) = a,
sqrt(det(g_{T^{3,-1}} restricted to Sigma_2)) = -c.
```

The parameter `nu` is extrinsic.  It is not determined by the metric on `Sigma_-`.  It is the one-dimensional free datum left when `mu = -1`; equivalently,

```text
nu = y3'(0).
```

Geometrically, `nu` controls the normal derivative of the induced singular-orbit metric, hence the second fundamental form in the radial normal direction:

```text
II_{partial_t}(X,Y) = (1/2) * d/dt(g_{Sigma_-})(X,Y).
```

So `nu` does not change the singular-orbit metric itself.  It changes how the singular orbit is embedded in the seven-manifold.  It is also the leading coefficient of the symmetry-breaking mode away from the `U(1)`-enhanced two-parameter subfamily.  When `nu = 0`, the solution lies in the extra-symmetric branch with

```text
q2 = q3,
q6 = q7,
p2 = p3
```

near the singular orbit, and in fact throughout the corresponding symmetric solution.

## 5. Implementation checklist

1. Fix `lambda`.
2. Choose `a > 0`, `c < 0`, and `nu`.
3. Compute `D`, `m`, `r`, `s`, `u`, `K`, and the `b_i` above.
4. Choose a small `eps` and initialize `q(eps)` using the Taylor data.
5. At each ODE step, compute `p1,p2,p3` from the algebraic constraints using the displayed sign branch.
6. Evaluate the eight-equation `q`-system.
7. Monitor the discriminant identity and metric positive-definiteness as diagnostics.
