/* eslint-disable no-console */

const { GetObjectCommand, ListObjectsV2Command, S3 } = require('@aws-sdk/client-s3');
const { ungzip } = require('node-gzip');
const { Pool } = require('pg');

const DATASET_BUCKET = 'neuronpedia-datasets';
const DATASET_REGION = 'us-east-1';
const DATASET_BASE_PATH = 'v1/';
const WORK_MEM = '2GB';

const modelId = 'gemma-2-2b';
const sourceSetName = 'gemmascope-transcoder-16k';

const s3 = new S3({
  region: DATASET_REGION,
  credentials: { accessKeyId: '', secretAccessKey: '' },
  signer: { sign: async (req) => req },
});

async function listAllActivationFiles(prefix) {
  let continuationToken;
  const files = [];

  while (true) {
    const response = await s3.send(
      new ListObjectsV2Command({
        Bucket: DATASET_BUCKET,
        Prefix: prefix,
        ContinuationToken: continuationToken,
      }),
    );

    for (const item of response.Contents || []) {
      const key = item.Key || '';
      if (!key.endsWith('.jsonl.gz')) continue;
      const relativePath = key.substring(prefix.length).replace(/^\/+/, '');
      if (relativePath.includes('/')) continue;
      files.push(key);
    }

    if (!response.IsTruncated) break;
    continuationToken = response.NextContinuationToken;
  }

  files.sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
  return files;
}

async function downloadAndDecompressFile(path) {
  const response = await s3.send(
    new GetObjectCommand({
      Bucket: DATASET_BUCKET,
      Key: path,
    }),
  );
  const bytes = await response.Body?.transformToByteArray();
  if (!bytes) return '';
  return (await ungzip(Buffer.from(bytes))).toString();
}

async function importJsonlString(tableName, jsonlData) {
  let pool;
  let client;
  jsonlData = jsonlData.replaceAll('\\u0000', ' ');

  try {
    pool = new Pool({
      connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
      ssl: false,
    });
    client = await pool.connect();
    await client.query(`SET work_mem = '${WORK_MEM}'`);

    const firstLine = jsonlData.trim().split('\n')[0];
    if (!firstLine) return 0;

    const availableColumns = Object.keys(JSON.parse(firstLine));
    const columnQuery = `
      SELECT
        column_name,
        CASE
          WHEN data_type = 'ARRAY' THEN udt_name::regtype::text || '[]'
          WHEN data_type = 'USER-DEFINED' THEN quote_ident(c.udt_name)
          ELSE data_type
        END AS data_type
      FROM information_schema.columns c
      WHERE table_name = $1
        AND column_name = ANY($2)
      ORDER BY ordinal_position
    `;
    const { rows: columns } = await client.query(columnQuery, [tableName, availableColumns]);

    const columnDefs = columns.map((col) => `"${col.column_name}" ${col.data_type}`).join(', ');
    const columnList = columns.map((col) => `"${col.column_name}"`).join(', ');
    const query = `
      INSERT INTO "${tableName}" (${columnList})
      SELECT ${columnList} FROM jsonb_to_recordset($1::jsonb) AS t(${columnDefs})
      ON CONFLICT DO NOTHING
    `;

    const lines = jsonlData.trim().split('\n');
    const chunkSize = 65000;
    let importedLines = 0;

    for (let i = 0; i < lines.length; i += chunkSize) {
      const chunk = lines.slice(i, i + chunkSize);
      const jsonArray = `[${chunk.join(',')}]`;
      await client.query(query, [jsonArray]);
      importedLines += chunk.length;
    }

    return importedLines;
  } finally {
    if (client) client.release();
    if (pool) await pool.end();
  }
}

async function getSourceIds() {
  const pool = new Pool({
    connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
    ssl: false,
  });
  const client = await pool.connect();
  try {
    const result = await client.query(
      `
        SELECT id
        FROM "Source"
        WHERE "modelId" = $1
          AND "setName" = $2
        ORDER BY id
      `,
      [modelId, sourceSetName],
    );
    return result.rows.map((row) => row.id);
  } finally {
    client.release();
    await pool.end();
  }
}

async function getActivationCount() {
  const pool = new Pool({
    connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
    ssl: false,
  });
  const client = await pool.connect();
  try {
    const result = await client.query(`SELECT count(*)::bigint AS count FROM "Activation" WHERE "modelId" = $1`, [
      modelId,
    ]);
    return result.rows[0].count;
  } finally {
    client.release();
    await pool.end();
  }
}

async function main() {
  if (!process.env.POSTGRES_URL_NON_POOLING) {
    throw new Error('POSTGRES_URL_NON_POOLING is not set');
  }

  const sourceIds = await getSourceIds();
  console.log(`Importing activations for ${modelId} / ${sourceSetName}`);
  console.log(`Found ${sourceIds.length} sources`);

  for (const sourceId of sourceIds) {
    const prefix = `${DATASET_BASE_PATH}${modelId}/${sourceId}/activations`;
    const files = await listAllActivationFiles(prefix);
    console.log(`[${sourceId}] ${files.length} activation files`);

    for (const [index, file] of files.entries()) {
      const startedAt = Date.now();
      console.log(`[${sourceId}] (${index + 1}/${files.length}) downloading ${file}`);
      const jsonl = await downloadAndDecompressFile(file);
      const imported = await importJsonlString('Activation', jsonl);
      const seconds = ((Date.now() - startedAt) / 1000).toFixed(1);
      console.log(`[${sourceId}] imported ${imported} rows from ${file} in ${seconds}s`);
    }
  }

  const finalCount = await getActivationCount();
  console.log(`Done. Activation rows for ${modelId}: ${finalCount}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
