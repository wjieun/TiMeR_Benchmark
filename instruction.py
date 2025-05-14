FINETUNE_INST = """
Resolve all temporal deixis in CONVERSATION according to the SPEECH TIME.
- Use ISO 8601 formatting. The default format is %Y-%m-%dT%H:%M.
- Weeks start on Monday and end on Sunday.
- "this day" refers to the same weekday of the current week.
- "last day" refers to the same weekday of the previous week.
- "next day" refers to the same weekday of the following week.
- If the SPEECH TIME is Saturday or Sunday, "this weekend" still refers to the current weekend.
- Do not resolve expressions referring to general or habitual time. (e.g., "August is hot", "I leave at 5 on Tuesdays")

| Temporal Unit | Format                | Example              |
|---------------|-----------------------|----------------------|
| Year          | YYYY                  | 2025                 |
| Quarter       | YYYY-Qn               | 2025-Q1              |
| Season        | YYYY-SeasonName       | 2025-Winter          |
| Month         | YYYY-MM               | 2025-01              |
| Week          | YYYY-Www              | 2025-W01             |
| Weekend       | YYYY-Www-WE           | 2025-W01-WE          |
| Day           | YYYY-MM-DD            | 2025-01-01           |
| Daypart       | YYYY-MM-DDTPP         | 2025-01-01TMO        |
| Datetime      | YYYY-MM-DDTHH:MM      | 2025-01-01T14:30     |
| Time          | HH:MM                 | 14:30                |

"""