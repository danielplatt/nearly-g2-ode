# Berger nearly $G_2$ metric: initial data and numerical target conditions

This note records the left-end initial data and the numerical success criteria for recovering the homogeneous Berger nearly $G_2$ solution using the $q_i$-system.

## 1. Which system to solve near $t=0$

Do **not** start the raw $q_i$-system exactly at $t=0$. In the $q_i$-equations, the denominators involve $p_1 p_2 p_3$, and one collapsing factor vanishes linearly at each singular orbit, so the raw system is singular at the endpoints.

Instead, start from the regularized left-end expansion and then evaluate the Taylor series at a small $t = \varepsilon > 0$.

For the Berger branch, with the normalization $\|dt\| = 1$, one takes

$$
\lambda = \frac{6}{\sqrt{5}}.
$$

If one reverses the orientation of $\varphi$, then $\lambda$ changes sign.

The cohomogeneity-one picture is:

- principal orbit $SO(4)/\mathbf Z_2^2$,
- singular orbit $SO(4)/O(2)$ at $t=0$ with weights $(3,1)$,
- singular orbit $SO(4)/O(2)$ at $t=\pi/3$ with weights $(1,3)$.

## 2. Left-end regularized expansion

Use the regularized variables written in the form

$$
\begin{aligned}
q_1 &= a + t^2 y_1, \\
q_2 &= t y_2, \\
q_3 &= t y_3, \\
q_4 &= c + t^2 y_4, \\
q_5 &= -3a + t^2 y_5, \\
q_6 &= t y_6, \\
q_7 &= t y_7, \\
q_8 &= -3c + t^2 y_8.
\end{aligned}
$$

For the Berger solution, the correct left-end parameters are

$$
a = \frac{\sqrt{5}}{20},
\qquad
c = -\frac{3\sqrt{5}}{100},
\qquad
\mu = -1.
$$

The zeroth-order regularized data are

$$
y(0)
=
\left(
\frac{9\sqrt{5}}{100},
\frac{\sqrt{15}}{25},
\frac{\sqrt{15}}{25},
\frac{\sqrt{5}}{100},
\frac{23\sqrt{5}}{100},
\frac{2\sqrt{15}}{25},
\frac{2\sqrt{15}}{25},
-\frac{9\sqrt{5}}{100}
\right).
$$

There is still one free odd coefficient locally. For the homogeneous Berger solution, it is

$$
y_2'(0) = -\frac{\sqrt{5}}{50},
\qquad
y_3'(0) = +\frac{\sqrt{5}}{50},
\qquad
y_6'(0) = -\frac{7\sqrt{5}}{50},
\qquad
y_7'(0) = +\frac{7\sqrt{5}}{50}.
$$

## 3. Taylor data for starting the raw solver at $t=\varepsilon$

If one wants to start the raw $q_i$-solver at a small $t=\varepsilon$, use

$$
\begin{aligned}
q_1(t) &= \frac{\sqrt{5}}{20} + \frac{9\sqrt{5}}{100} t^2 + O(t^4), \\
q_2(t) &= \frac{\sqrt{15}}{25} t - \frac{\sqrt{5}}{50} t^2 + O(t^3), \\
q_3(t) &= \frac{\sqrt{15}}{25} t + \frac{\sqrt{5}}{50} t^2 + O(t^3), \\
q_4(t) &= -\frac{3\sqrt{5}}{100} + \frac{\sqrt{5}}{100} t^2 + O(t^4), \\
q_5(t) &= -\frac{3\sqrt{5}}{20} + \frac{23\sqrt{5}}{100} t^2 + O(t^4), \\
q_6(t) &= \frac{2\sqrt{15}}{25} t - \frac{7\sqrt{5}}{50} t^2 + O(t^3), \\
q_7(t) &= \frac{2\sqrt{15}}{25} t + \frac{7\sqrt{5}}{50} t^2 + O(t^3), \\
q_8(t) &= \frac{9\sqrt{5}}{100} - \frac{9\sqrt{5}}{100} t^2 + O(t^4).
\end{aligned}
$$

Equivalently, the formal endpoint values are

$$
q(0)
=
\left(
\frac{\sqrt{5}}{20},
0,
0,
-\frac{3\sqrt{5}}{100},
-\frac{3\sqrt{5}}{20},
0,
0,
\frac{9\sqrt{5}}{100}
\right),
$$

but one should **not** start the singular ODE exactly there.

## 4. What counts as numerical success?

There are two practical tests.

### 4.1 Interior test at $t = \pi/6$

There is no smoothness condition at $t = \pi/6$.

What is special for the homogeneous Berger metric is that $t = \pi/6$ is the unique maximal-volume principal orbit. Hence the clean interior condition is

$$
l\!\left(\frac{\pi}{6}\right) = 0,
$$

or equivalently

$$
\frac{d}{dt}(p_1 p_2 p_3)\Big|_{t=\pi/6} = 0.
$$

In the explicit homogeneous formula, the orbit-volume density is proportional to

$$
V(t) \propto \sin t\,(4\cos^2 t - 1),
$$

and therefore

$$
V'\!\left(\frac{\pi}{6}\right) = 0.
$$

So, if one is shooting in the free parameter $y_2'(0)$, then the simplest interior target is

$$
l\!\left(\frac{\pi}{6}\right) = 0.
$$

### 4.2 Closing test at $t = \pi/3$

For the right-hand endpoint, set

$$
s = \frac{\pi}{3} - t.
$$

For Berger, the right-end singular-orbit parameters are

$$
d = -\frac{\sqrt{5}}{20} = -a,
\qquad
f = \frac{3\sqrt{5}}{100} = -c.
$$

Thus the solution must satisfy the asymptotics

$$
q_1 = 3f + O(s^2),
\qquad
q_2 = f + O(s^2),
\qquad
q_7 = -3d + O(s^2),
\qquad
q_8 = -d + O(s^2),
$$

and also

$$
q_3, q_4, q_5, q_6 = O(s),
$$

with the sharper even/odd relations

$$
q_3 + q_5 = O(s^2),
\qquad
q_4 + q_6 = O(s^2).
$$

Numerically, this means that at the endpoint one should see

$$
q\!\left(\frac{\pi}{3}\right)
=
\left(
\frac{9\sqrt{5}}{100},
\frac{3\sqrt{5}}{100},
0,
0,
0,
0,
\frac{3\sqrt{5}}{20},
\frac{\sqrt{5}}{20}
\right)
$$

in these conventions, together with the residuals

$$
q_1 - 3 q_2,
\qquad
q_7 - 3 q_8,
\qquad
q_3 + q_5,
\qquad
q_4 + q_6
$$

vanishing quadratically in $s$, while

$$
q_3,
\quad q_4,
\quad q_5,
\quad q_6
$$

vanish linearly in $s$.

This is the true right-end closing condition. It reflects the switch of collapsing weight from the left end to the right end.

## 5. Practical numerical recipe

1. Fix the Berger branch
   $$
   \lambda = \frac{6}{\sqrt{5}},
   \qquad
   a = \frac{\sqrt{5}}{20},
   \qquad
   c = -\frac{3\sqrt{5}}{100},
   \qquad
   y_2'(0) = -\frac{\sqrt{5}}{50}.
   $$

2. Evaluate the Taylor series above at a small $t=\varepsilon$ and use that as initial data for the raw $q_i$-ODE solver.

3. Integrate on $[\varepsilon, \pi/3 - \varepsilon]$.

4. Check the interior condition
   $$
   l\!\left(\frac{\pi}{6}\right) \approx 0.
   $$

5. Check the right-end closing conditions near $t=\pi/3$:
   - $q_1 \approx 3 q_2$,
   - $q_7 \approx 3 q_8$,
   - $q_3, q_4, q_5, q_6 \approx 0$,
   - $q_3 + q_5 \approx 0$ to higher order,
   - $q_4 + q_6 \approx 0$ to higher order.

6. Also verify that no unexpected denominator vanishes on $(0, \pi/3)$.

If these conditions hold, then numerically one has found the Berger nearly $G_2$ solution.
