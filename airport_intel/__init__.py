"""Airport Investment Intelligence Agent — core package.

Layout:
  agent       orchestration: route -> execute -> revise -> narrate
  tools       deterministic tools the agent calls (ranking, compare, profile, ...)
  scoring     the Expansion Profitability Index (EPI) engine + bounded LLM modifier
  repository  data-access layer (Repository interface; JSON-backed today)
  regions     US region -> state mapping + resolution
  llm         swappable LLM providers (Gemini | Claude) behind one `complete` primitive
"""
