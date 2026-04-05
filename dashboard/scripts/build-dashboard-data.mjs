import fs from 'fs';
import path from 'path';

import {
  ANNOTATIONS_PATH,
  PRECOMPUTED_DASHBOARD_DATA_PATH,
  buildGoldenSetDashboardDataFromRaw,
} from '../src/lib/golden-set.ts';

const outputDir = path.dirname(PRECOMPUTED_DASHBOARD_DATA_PATH);

if (!fs.existsSync(ANNOTATIONS_PATH)) {
  console.error(`Missing raw annotations file: ${ANNOTATIONS_PATH}`);
  process.exit(1);
}

fs.mkdirSync(outputDir, { recursive: true });

const data = buildGoldenSetDashboardDataFromRaw();
fs.writeFileSync(PRECOMPUTED_DASHBOARD_DATA_PATH, `${JSON.stringify(data)}\n`, 'utf8');

const stats = fs.statSync(PRECOMPUTED_DASHBOARD_DATA_PATH);
console.log(`Wrote ${PRECOMPUTED_DASHBOARD_DATA_PATH} (${stats.size} bytes)`);
