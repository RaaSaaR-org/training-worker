"""Evaluation-stage job runners (reward models, dataset annotation).

These subclass ``trainers.base.BaseTrainer`` so they reuse the whole
worker job plumbing (claim → heartbeat → progress → artifact → complete)
even though they evaluate/annotate rather than train.
"""
