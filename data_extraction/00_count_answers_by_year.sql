-- Count answers by year. This aggregate fits in the SEDE web UI
-- (50,000-row cap) and is the check against a local dump, not the extract.

SELECT
    DATEPART(year, CreationDate) AS answer_year,
    COUNT(*) AS answers
FROM Posts
WHERE PostTypeId = 2
GROUP BY DATEPART(year, CreationDate)
ORDER BY answer_year;
