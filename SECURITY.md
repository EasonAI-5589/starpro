# Security policy

Do not report exposed credentials or other sensitive information in a public
issue. Use GitHub's private vulnerability reporting for this repository so the
maintainers can investigate without disclosing the report.

Treat any credential that has appeared in a public commit as compromised:
revoke or rotate it at the provider first, then remove it from the current tree.
History rewriting, when necessary, is handled as a separate coordinated step
because it changes commit identifiers for every contributor.
