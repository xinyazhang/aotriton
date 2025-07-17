UPDATE Primary SET
    status = 1,                 -- ASSIGNED
    client = ?,
    ip = ? 
WHERE
    status = 0,                 -- IDLE
RETURNING (id, jdesc)
LIMIT ?
