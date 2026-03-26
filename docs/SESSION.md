# Session — 2026-03-25 (continued)

## Goal
Analyze R2-Reasoner paper (arXiv:2506.05901v2) for ideas applicable to Crystal's routing system.

## Notes
- R2-Reasoner decomposes queries into subtasks, then allocates each subtask to a different-capability model
- Crystal currently has a fixed routing pipeline: detect → plan → tool execution → compile → route (direct return vs LLM)
- Key paper ideas with Crystal relevance: subtask decomposition, difficulty-based model selection, grouped search for optimal allocation, cost-accuracy Pareto tradeoffs
