-- Body, title, and tag edits. PostHistoryTypeId 5 is an edit to the body.
-- The automation rule's "no subsequent editing activity" uses
-- Posts.LastEditDate in the cleaner. This extract supports an audit of
-- that field. The manuscript does not state the horizon used to code
-- future edits, so this query does not collapse edits into one horizon.

SELECT
    ph.PostId       AS answer_id,
    ph.CreationDate AS edit_date,
    ph.UserId       AS editor_user_id,
    ph.PostHistoryTypeId AS history_type_id
FROM PostHistory AS ph
INNER JOIN Posts AS a
    ON a.Id = ph.PostId
   AND a.PostTypeId = 2
WHERE ph.PostHistoryTypeId IN (4, 5, 6);
