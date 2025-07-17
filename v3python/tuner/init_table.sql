CREATE TABLE Primary (          -- Intentionally without "IF NOT EXISTS"
    id INTEGER,
    jdesc TEXT,                 -- json description
    status INTEGER,             -- check job.PrimaryStatus
    client TEXT,                -- assigned to client
    ip TEXT,                    -- ip of client
    PRIMARY KEY (id)
);
CREATE TABLE Secondary (
    FOREIGN KEY (primary) REFERENCES Primary(id),   -- No "on delete", Primary(id, jdesc) is immutable once added
    subkernel TEXT,
    kernel_no INTEGER,
    status INTEGER,                                 -- check job.SecondaryStatus
    jreport TEXT,                                   -- json report
    PRIMARY KEY (primary, subkernel, kernel_no) 
);
