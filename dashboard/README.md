# AI Risk Observatory — Dashboard

Next.js visualization app for the [AI Risk Observatory](../README.md). Displays AI risk, adoption, and third-party exposure signals across UK public-company annual reports.

## Data workflow

The deployed dashboard reads a precomputed artifact at `data/dashboard-data.json`.

To refresh local dashboard data from the raw pipeline outputs and regenerate the artifact:
```bash
npm run sync:data
```

If the raw files are already present in `data/`, regenerate only the artifact:
```bash
npm run build:data
```

## Development

```bash
npm install
npm run dev
# open http://localhost:3000
```

## Deployment

Deployed on Vercel. The `data/dashboard-data.json` artifact is committed and served statically — no database or API calls at runtime.
