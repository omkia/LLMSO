-- Answer extract for a local restoration of the Stack Exchange Data Dump
-- in the SEDE relational schema (Posts, Users).
--
-- The manuscript's raw file contains 5,847,312 answers, including answers
-- outside January 2019-December 2025. Those exclusions are applied in
-- preprocessing so each stage can be counted. This query does not embed
-- that row count: it is a property of one dump vintage.
--
-- The SEDE website cannot return this result. It caps a query at 50,000
-- rows. Run this against a restored dump.

SELECT
    a.Id                    AS answer_id,
    a.ParentId              AS question_id,
    a.OwnerUserId           AS user_id,
    q.OwnerUserId           AS question_owner_id,
    a.CreationDate          AS answer_creation_date,
    u.CreationDate          AS account_creation_date,
    a.Body                  AS answer_body,
    a.Score                 AS net_vote,
    a.CommentCount          AS comment_count,
    a.CommunityOwnedDate    AS community_owned_date,
    a.LastEditDate          AS answer_last_edit_date,
    q.CreationDate          AS question_creation_date,
    q.Body                  AS question_body,
    q.Tags                  AS tags,
    q.AcceptedAnswerId      AS accepted_answer_id,
    q.ClosedDate            AS closed_date,
    CASE
        WHEN a.OwnerUserId = q.OwnerUserId THEN 1
        ELSE 0
    END                     AS self_answer
FROM Posts AS a
INNER JOIN Posts AS q
    ON q.Id = a.ParentId
   AND q.PostTypeId = 1
LEFT JOIN Users AS u
    ON u.Id = a.OwnerUserId
WHERE a.PostTypeId = 2;
