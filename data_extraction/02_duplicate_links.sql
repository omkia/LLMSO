-- Community duplicate links. LinkTypeId 3 is Duplicate.
-- The manuscript keeps these questions and stores a binary indicator.
-- It does not drop them.

SELECT
    PostId          AS question_id,
    RelatedPostId   AS related_question_id,
    CreationDate    AS link_creation_date,
    LinkTypeId      AS link_type_id
FROM PostLinks
WHERE LinkTypeId = 3;
