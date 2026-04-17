# Handover: numerical exploration of the 8-function Einstein ODE system

## 1. Goal

We want an exploratory numerical solver for a singular ODE system for
\(y_1,\dots,y_8\) on \([0,\pi/3]\), arising from a cohomogeneity-one Einstein metric problem in dimension 7.

For the **first implementation**, the aim is **not** a rigorous proof. The goal is:

1. construct a high-accuracy numerical solution from the singular end \(t=0\),
2. march it up to the symmetry point \(t=\pi/6\),
3. evaluate the midpoint shooting defect,
4. do this as a function of the single shooting parameter \(\alpha\).

The expected exact reflective symmetry is about \(\pi/6\), and for the first implementation we assume the reflection acts trivially on the chosen variables, i.e. \(S=\mathrm{Id}\). Under this assumption, the midpoint target is
\[
y'\!\left(\frac\pi6\right)=0.
\]

So numerically we should shoot from \(t=0\) to \(t=\pi/6\) and monitor
\[
\Phi(\alpha):=y'\!\left(\frac\pi6;\alpha\right)\in\mathbb R^8.
\]

For now, the code only needs to evaluate \(\Phi(\alpha)\) for a supplied \(\alpha\). Root-finding in \(\alpha\) can be added later.

---

## 2. Fixed parameters

Use
\[
a=\frac{\sqrt5}{20},\qquad c=-\frac{3\sqrt5}{100},\qquad \lambda=\frac6{\sqrt5},\qquad \mu_1=\mu_2=1.
\]

All square roots are taken on the **positive real branch**.

---

## 3. The ODE system

Define
\[
Q=-(y_2+y_7)(y_3+y_6)\bigl((c-3a)+t^2(y_4+y_5)\bigr).
\]

Then the system is

\[
\begin{aligned}
y_1'(t)
&=\frac1t\Biggl(
-2y_1
-\frac{a\lambda^{5/2}}{2\sqrt Q}\Bigl(a(3y_4+y_8)-c(3y_1+y_5)-y_2y_7-y_3y_6-6y_2y_3\Bigr) \\
&\qquad\qquad
-\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
a(y_1y_8-y_4y_5)
+y_1\bigl(a(3y_4+y_8)-c(3y_1+y_5)-y_2y_7-y_3y_6\bigr)
+2y_2y_3y_5 \\
&\qquad\qquad\qquad\qquad\qquad
+t^2\bigl(y_1^2y_8-y_1y_4y_5\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_2'(t)
&=\frac1t\Biggl(
-y_2
+\sqrt{\frac{-(y_3+y_6)\bigl(c-3a+t^2(y_4+y_5)\bigr)}{\lambda(y_2+y_7)}}
+\frac{\lambda^{5/2}ac}{\sqrt Q}(3y_2+y_6) \\
&\qquad\qquad
+\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
y_2^2y_7-y_2y_3y_6
-a(y_2y_8-3y_2y_4-2y_4y_6)
-c(y_2y_5-3y_1y_2-2y_1y_6) \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_2y_8+y_2y_4y_5-2y_1y_4y_6\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_3'(t)
&=\frac1t\Biggl(
-y_3
+\sqrt{\frac{-(y_2+y_7)\bigl(c-3a+t^2(y_4+y_5)\bigr)}{\lambda(y_3+y_6)}}
-\frac{\lambda^{5/2}ac}{\sqrt Q}(3y_3+y_7) \\
&\qquad\qquad
+\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
y_3^2y_6-y_2y_3y_7
-a(y_3y_8-3y_3y_4-2y_4y_7)
-c(y_3y_5-3y_1y_3-2y_1y_7) \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_3y_8+y_3y_4y_5-2y_1y_4y_7\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_4'(t)
&=\frac1t\Biggl(
-2y_4
+\sqrt{\frac{-(y_2+y_7)(y_3+y_6)}{\lambda\bigl(c-3a+t^2(y_4+y_5)\bigr)}}
+\frac{c\lambda^{5/2}}{2\sqrt Q}\Bigl(a(3y_4+y_8)-c(3y_1+y_5)+y_2y_7+y_3y_6+6y_2y_3\Bigr) \\
&\qquad\qquad
-\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
c(y_4y_5-y_1y_8)
+y_4\bigl(-a(3y_4+y_8)+c(3y_1+y_5)-y_2y_7-y_3y_6\bigr)
+2y_2y_3y_8 \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_4y_8-y_4^2y_5\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_5'(t)
&=\frac1t\Biggl(
-2y_5
-\sqrt{\frac{-(y_2+y_7)(y_3+y_6)}{\lambda\bigl(c-3a+t^2(y_4+y_5)\bigr)}}
+\frac{a\lambda^{5/2}}{2\sqrt Q}\Bigl(3a(3y_4+y_8)-3c(3y_1+y_5)+3y_2y_7+3y_3y_6+2y_6y_7\Bigr) \\
&\qquad\qquad
+\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
-3a(y_4y_5-y_1y_8)
+y_5\bigl(-a(3y_4+y_8)+c(3y_1+y_5)-y_2y_7-y_3y_6\bigr)
+2y_1y_6y_7 \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_5y_8-y_4y_5^2\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_6'(t)
&=\frac1t\Biggl(
-y_6
-\sqrt{\frac{-(y_2+y_7)\bigl(c-3a+t^2(y_4+y_5)\bigr)}{\lambda(y_3+y_6)}}
-\frac{3\lambda^{5/2}ac}{\sqrt Q}(y_6+3y_2) \\
&\qquad\qquad
-\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
y_3y_6^2-y_2y_6y_7
-a(y_6y_8-3y_4y_6+6y_2y_8)
-c(y_5y_6-3y_1y_6+6y_2y_5) \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_6y_8+y_4y_5y_6-2y_2y_5y_8\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_7'(t)
&=\frac1t\Biggl(
-y_7
-\sqrt{\frac{-(y_3+y_6)\bigl(c-3a+t^2(y_4+y_5)\bigr)}{\lambda(y_2+y_7)}}
+\frac{3\lambda^{5/2}ac}{\sqrt Q}(3y_3+y_7) \\
&\qquad\qquad
-\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
y_2y_7^2-y_3y_6y_7
-a(y_7y_8-3y_4y_7+6y_3y_8)
-c(y_5y_7-3y_1y_7+6y_3y_5) \\
&\qquad\qquad\qquad\qquad\qquad
-t^2\bigl(y_1y_7y_8+y_4y_5y_7-2y_3y_5y_8\bigr)
\Bigr)
\Biggr),
\end{aligned}
\]

\[
\begin{aligned}
y_8'(t)
&=\frac1t\Biggl(
-2y_8
+\frac{c\lambda^{5/2}}{2\sqrt Q}\Bigl(-3a(3y_4+y_8)+3c(3y_1+y_5)+3(y_2y_7+y_3y_6)+2y_6y_7\Bigr) \\
&\qquad\qquad
+\frac{\lambda^{5/2}}{2\sqrt Q}\,t\Bigl(
-3c(y_1y_8-y_4y_5)
+y_8\bigl(a(3y_4+y_8)-c(3y_1+y_5)-y_2y_7-y_3y_6\bigr)
+2y_4y_6y_7 \\
&\qquad\qquad\qquad\qquad\qquad
+t^2\bigl(y_1y_8^2-y_4y_5y_8\bigr)
\Bigr)
\Biggr).
\end{aligned}
\]

The right-hand side is singular at \(t=0\), but the smooth branch is expected to exist and be uniquely determined locally by the zeroth-order data plus one scalar first-order parameter \(\alpha\).

---

## 4. Initial data at \(t=0\)

Define
\[
\xi:=y(0).
\]

For the fixed parameters above, the initial values simplify to
\[
\xi=
\left(
\frac{9\sqrt5}{100},
\frac{4\sqrt{15}}{25},
\frac{4\sqrt{15}}{25},
\frac{\sqrt5}{100},
\frac{83\sqrt5}{100},
-\frac{7\sqrt{15}}{25},
-\frac{7\sqrt{15}}{25},
-\frac{9\sqrt5}{20}
\right).
\]

The single free first-order parameter is
\[
\alpha:=y_2'(0).
\]

The remaining first derivatives are determined by
\[
y_1'(0)=y_4'(0)=y_5'(0)=y_8'(0)=0,
\]
\[
y_3'(0)=-\alpha,
\qquad
\ y_6'(0)=7\alpha,
\qquad
\ y_7'(0)=-7\alpha.
\]

So
\[
y'(0)=\alpha\,\rho,
\qquad
\rho:=(0,1,-1,0,0,7,-7,0).
\]

---

## 5. Local ansatz at the singular end

Use the regularised ansatz
\[
y(t)=\xi+\alpha t\,\rho+t^2 z(t),
\]
where \(z(t)\) is expected to be smooth near \(t=0\). Equivalently,
\[
y(t)=\sum_{n\ge0} a_n t^n,
\qquad
 a_0=\xi,
\qquad
 a_1=\alpha\rho.
\]

For the first implementation, use a **Frobenius/power-series solver** near \(t=0\): plug a truncated Taylor polynomial into the ODE, expand formally in powers of \(t\), compare coefficients, and solve recursively for the unknown higher-order coefficients.

The leading exponent for the smooth branch is \(1\): there is no fractional-power ansatz in the current variables.

---

## 6. Numerical strategy

### 6.1 Overview

Use **arbitrary-precision arithmetic** and **piecewise Taylor marching**.

1. Compute a high-order local Taylor series at \(t=0\) from the ansatz above.
2. Evaluate that series at a nearby point \(t_1>0\) within its apparent convergence radius.
3. Re-expand about \(t_1\) using another local Taylor series.
4. Repeat until reaching \(t=\pi/6\).
5. Evaluate the midpoint defect \(y'(\pi/6)\).

### 6.2 Important warning near \(t=0\)

Do **not** numerically evaluate the raw formula
\[
y'=\frac1t F(t,y)
\]
very close to \(t=0\) using floating-point substitution. Smoothness at \(t=0\) is encoded by delicate cancellation in the numerator. The correct approach is to enforce the local Taylor/Frobenius structure from the outset.

### 6.3 Arithmetic

Use arbitrary precision from the start.

Recommended first choice: `python-flint` / FLINT ball arithmetic, since this is also what Buttsworth–Hodgkinson use in their `O3O10` code. Exact symbolic preprocessing is fine, but actual marching should be done at high precision rather than in machine doubles.

### 6.4 Formal Taylor solver

At each expansion centre \(t_0\), write
\[
y(t)=\sum_{n=0}^N c_n (t-t_0)^n.
\]

At \(t_0=0\), we have
\[
c_0=\xi,
\qquad
 c_1=\alpha\rho.
\]

Then substitute into the ODE and compare coefficients of powers of \(t-t_0\).

Important: the square roots and rational expressions should themselves be expanded as formal series in \((t-t_0)\), rather than re-evaluated from the raw closed formulas at each order.

### 6.5 Marching / re-expansion

After obtaining a local series at \(t_0\):

1. estimate a local radius of convergence heuristically from the tail coefficients,
2. move forward by a fixed fraction of that radius (e.g. half the estimate),
3. re-expand at the new centre.

This is the same broad pattern as in the Buttsworth–Hodgkinson code.

### 6.6 Output for the first version

Given \(\alpha\), the code should return at least:

- the computed local series data,
- the list of expansion centres,
- the numerical approximation to \(y(\pi/6)\),
- the numerical approximation to \(y'(\pi/6)\),
- diagnostics tracking the branch/invariant conditions below.

No root finder in \(\alpha\) is required in the first version.

---

## 7. Runtime invariants / branch conditions

Along the desired real solution, we expect to remain in the region
\[
Q>0,
\qquad
 y_2+y_7<0,
\qquad
 y_3+y_6<0,
\qquad
 c-3a+t^2(y_4+y_5)<0.
\]

These conditions matter because they keep the square roots on the chosen real branch and prevent denominator blow-up.

They should be monitored continuously (or at least at all expansion centres and several interior sample points per step). Any violation should be treated as a hard diagnostic failure.

At \(t=0\), one has
\[
Q(0)=\frac{243\sqrt5}{6250}>0.
\]

---

## 8. Midpoint target

For the first implementation, assume the reflection is literally
\[
y(t)=y\!\left(\frac\pi3-t\right)
\]
componentwise.

Under this assumption,
\[
y'\!\left(\frac\pi6\right)=0
\]
is the correct midpoint condition, and it is preferable numerically to shooting all the way to \(\pi/3\).

So the mismatch map for exploration is
\[
\Phi(\alpha)=y'\!\left(\frac\pi6;\alpha\right).
\]

Later, if the reflection acts nontrivially on the variables, this condition will have to be modified.

---

## 9. Sanity checks the code should perform automatically

1. Verify that the local series satisfies
   \[
   y(0)=\xi,
   \qquad
   y'(0)=\alpha\rho.
   \]
2. Verify the first-derivative relations componentwise.
3. Check that the recursively computed Taylor coefficients satisfy the ODE to the expected order.
4. Monitor the branch conditions in Section 7.
5. Check numerical stability under increases of working precision and Taylor order.
6. Check stability of the midpoint defect \(\Phi(\alpha)\) under refinement of the marching parameters.

---

## 10. Notes from the Buttsworth–Hodgkinson implementation (`O3O10`)

The attached `O3O10` repository uses the following pattern.

1. `python-flint` / FLINT ball arithmetic, not machine doubles.
2. A power-series solver at the singular end (`Eta.initial_series`).
3. Piecewise re-expansion / Taylor marching away from the singular end (`Eta.next_series`).
4. Heuristic estimation of radius of convergence from tail coefficients.
5. Chebyshev fitting later, for proof-oriented polynomial representation.
6. Finite differences for derivatives with respect to the shooting parameter in `linearized.py`.

For the present 8-function system, items 1--4 are the relevant design pattern for the first implementation. Items 5--6 can be postponed.

---

## 11. Mathematical status / proof-level caveat

The exploratory code uses a power-series ansatz at \(t=0\). This is a **numerical choice**, not yet a proof-level assumption.

A final numerically verified proof need not work in an analytic function space: one could use Taylor marching only to generate an approximate solution, and then validate it in a Banach space by residual bounds plus a linear inverse estimate. So the exploratory numerical method does **not** commit the final proof strategy to analyticity.

---

## 12. Suggested first coding tasks

1. Implement the ODE right-hand side with arbitrary precision.
2. Implement the data \((a,c,\lambda,\xi,\rho)\).
3. Implement a formal Taylor-series engine at \(t=0\) with input \(\alpha\) and order \(N\).
4. Implement series evaluation and derivative evaluation.
5. Implement one re-expansion step about a general centre \(t_0\).
6. Implement repeated marching to \(\pi/6\).
7. Return the midpoint defect \(\Phi(\alpha)\) and branch diagnostics.
8. Add convergence/refinement tests in precision and Taylor order.

---

## 13. Implementation details to be decided

1. Whether to code directly with the unknown series for \(y\), or instead with the regularised unknown \(z\) in
   \[
   y(t)=\xi+\alpha t\rho+t^2 z(t).
   \]
   Coding directly for \(y\) is simpler; coding \(z\) is conceptually cleaner.

2. Exact software stack:
   - `python-flint` only,
   - `sympy` for symbolic preprocessing + `python-flint` for numerics,
   - or some other hybrid.

3. Exact policy for choosing Taylor order, working precision, and step size.

4. Whether to add Chebyshev compression at an early stage, or postpone that until/if a proof pipeline is attempted.

5. When and how to wrap the midpoint defect \(\Phi(\alpha)\) in a one-dimensional root finder.

