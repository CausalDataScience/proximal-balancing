# Single Proxy Fails, Single Proxy Plus Mediator Works

## Mechanism

The examples use two latent bits:

$$
U=(B,C).
$$

The single proxy is

$$
W=B.
$$

Thus \(W\) is a valid proxy but it cannot see the latent subtype \(C\). Treatment depends on \(B\) and \(C\), and the outcome depends on \(C\). Therefore exact conditioning or balancing on \(W\) still leaves residual hidden confounding.

The mediator is

$$
M=A\oplus C.
$$

So the mediator response reveals the missing subtype after the treatment arm is known:

$$
C=A\oplus M.
$$

This is the whole point of the DGP. The method is not ordinary post-treatment adjustment. It is a mediator-as-second-view construction: one external proxy \(W\) plus the response pattern \(M\) recovers the latent state \(U=(W,A\oplus M)\).

## Exact Numerical DGPs

| DGP | True ATE | W-only estimate | W+mediator estimate | Residual \(U\) imbalance given \(W\) | Residual \(U\) imbalance given \((W,A\oplus M)\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| subtype response | 0.520 | 1.007 | 0.520 | 0.625 | 0.000 |
| adherence response | 0.252 | 0.951 | 0.252 | 0.523 | 0.000 |
| resistance response | 0.884 | 1.052 | 0.884 | 0.674 | 0.000 |

## Real-World Example 1: Antiviral Treatment

- \(A\): assign an antiviral therapy.
- \(M\): early viral-load response after treatment assignment.
- \(Y\): later clinical outcome.
- \(B\): baseline inflammation level.
- \(C\): unmeasured resistance or immune-response subtype.
- \(W\): routine baseline liver panel, which captures \(B\) but not \(C\).

Balancing on the liver panel \(W\) does not remove confounding because treated and untreated patients can still differ in the hidden resistance subtype \(C\). However, the early mediator response \(M\) has different meaning by treatment arm: a response under treatment versus under no treatment reveals the hidden subtype. Thus \((W,A,M)\) can recover the latent state needed for the front-door functional.

## Real-World Example 2: Care-Management Intervention

- \(A\): assign a care-management intervention.
- \(M\): early adherence response, such as first-month medication pickup.
- \(Y\): hospitalization or disease-control outcome.
- \(B\): observed baseline engagement.
- \(C\): unmeasured adherence barrier.
- \(W\): pre-treatment engagement proxy, such as prior portal use.

The pre-treatment proxy \(W\) measures engagement but misses the hidden barrier \(C\). Treatment assignment may depend on clinician judgment of that barrier, so \(W\)-only balancing is still confounded. The early adherence response \(M\), interpreted together with assignment \(A\), reveals whether the patient had that barrier.

## Real-World Example 3: Online Recommendation

- \(A\): assign a recommendation or nudge.
- \(M\): immediate click or short-session response.
- \(Y\): long-run retention or conversion.
- \(B\): baseline activity level.
- \(C\): latent user intent.
- \(W\): pre-treatment activity proxy.

Balancing on \(W\) aligns baseline activity but not latent intent. The immediate response \(M\) under a known recommendation \(A\) reveals intent type, so the mediated structure supplies information that a single pre-treatment proxy cannot.

## Interpretation

The defensible statement is:

> A single proxy alone cannot generally deconfound treatment-outcome relations. A mediator can help when the mediator response is an additional observed view of the latent state and the causal estimand is a mediated effect.

The important caveat is that this requires a structural restriction on the mediator response. If \(M\) is merely another post-treatment variable and does not help resolve the missing part of \(U\), the rescue fails.
