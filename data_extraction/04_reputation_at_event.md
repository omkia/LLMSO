# Reputation on 30 November 2022

`Users.Reputation` in the data dump is reputation on the dump date, not on 30 November 2022. The manuscript's contributor-status variable is the standardized value of log(1 + reputation) on that date.

The Stack Exchange API method `GET /users/{ids}/reputation-history` returns reputation-change events. Reconstruct the level by ordering those events and summing through 30 November 2022. The public dump has no reputation-history table, and vote totals do not equal reputation: reputation also changes with accepted answers, bounties, suggested edits, and the association bonus.

Store the reconstructed level as `reputation_pre`. The analysis code then uses `status_pre = z(log(1 + reputation_pre))`, computed on the pre-period contributor file before any sample is drawn.

Questions used to prompt the synthetic detector class must be held out of the analytical sample. Pass that question-id list to `models/train_ai_detector.py`.
