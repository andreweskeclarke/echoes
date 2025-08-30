@README.md

After modifying the page you'll want to use the Playgwright MCP to take a screenshot and verify changes. The page at https://dashboard.lonel.ai/ has a user/password auth protection, so run things locally:
```
Bash(cd /var/www/dashboard && python -m http.server 8080 > /dev/null 2>&1 &)
```

Often I'll run that for you. Ask me or check using pgrep.