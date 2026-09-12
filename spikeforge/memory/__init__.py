"""SNN-native one-shot associative memory (issue #22).

A memory module built from ordinary LIF dynamics whose weights are
written by a single local Hebbian update instead of gradient descent,
so a class taught in one shot cannot disturb any other class's
weights. This is the reusable piece behind the held-out-digit proof
of concept and later applications (issues #23, #24).
"""
