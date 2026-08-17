                         START
                           │
                           ▼
                    ┌─────────────┐
                    │ ORCHESTRATOR│
                    │             │
                    │ Create plan │
                    └──────┬──────┘
                           │
                  5-7 tasks/sections
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Worker 1      Worker 2      Worker 3   ... Worker 7
             │             │             │
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                       REDUCER
                           │
                           ▼
                       FINAL BLOG
                           │
                           ▼
                          END
