import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = process.env.SPARKDASH_ROOT;
if (!source || !fs.existsSync(path.join(source, 'server/collectors/DecodeBench.js'))) {
  throw Error('Set SPARKDASH_ROOT to a local sparkDash checkout before running this benchmark.');
}
const benchHost = process.env.BENCH_HOST || '127.0.0.1';
const benchPort = Number(process.env.BENCH_PORT || 18890);
const modelId = process.env.BENCH_MODEL || 'glm-5.3-flash';
if (!Number.isInteger(benchPort) || benchPort < 1 || benchPort > 65535) {
  throw Error('BENCH_PORT must be an integer from 1 to 65535.');
}
const outputDir = process.env.BENCH_OUTPUT_DIR || path.join(here, 'local-results');
fs.mkdirSync(outputDir, { recursive: true });
process.env.BENCH_HISTORY_PATH = path.join(outputDir, 'decode-history.json');
process.env.BENCH_ACTIVE_PATH = path.join(outputDir, 'decode-active.json');
process.env.PREFILL_BENCH_HISTORY_PATH = path.join(outputDir, 'prefill-history.json');
process.env.PREFILL_BENCH_ACTIVE_PATH = path.join(outputDir, 'prefill-active.json');

const collector = file => pathToFileURL(path.join(source, 'server/collectors', file)).href;
const { decodeBenchManager } = await import(collector('DecodeBench.js'));
const { prefillBenchManager } = await import(collector('PrefillBench.js'));
const { closeLlmStreamAgent } = await import(collector('LlmStreaming.js'));

const result = {
  startedAt: new Date().toISOString(),
  harness: { name: 'sparkDash', version: '1.8.8', sourceCommit: '754f40a' },
  target: { url: `http://${benchHost}:${benchPort}`, via: 'local endpoint or SSH tunnel', model: modelId },
  protocol: { temperature: 0, topP: 1, thinking: false, decodeMaxTokens: 400, concurrencies: [1, 2, 3, 4], contextSizes: [8192, 16384, 32768, 65536, 131072, 262144] },
  jobs: {},
};
const out = path.join(outputDir, 'results.json');
const save = () => fs.writeFileSync(out, JSON.stringify(result, null, 2) + '\n');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function wait(manager, id, label) {
  let seen = -1;
  for (;;) {
    const job = manager.getJob(id);
    if (!job) throw Error(`${label}: job disappeared`);
    if (job.progress.completedLevels !== seen) {
      seen = job.progress.completedLevels;
      console.log(`${new Date().toISOString()} ${label}: ${seen}/${job.progress.totalLevels} ${job.progress.message}`);
      if (job.results?.length) {
        const row = job.results.at(-1);
        console.log(JSON.stringify(label === 'prefill' ? {
          targetTokens: row.targetTokens, promptTokens: row.promptTokens, prefillTps: row.prefillTps, ttftMs: row.ttftMs, error: row.error,
        } : {
          concurrency: row.concurrency, aggregateDecodeTps: row.aggregateDecodeTps, meanDecodeTps: row.meanDecodeTps, streamsOk: row.streamsOk, error: row.error,
        }));
      }
    }
    if (job.status !== 'running') return job;
    await pause(1000);
  }
}

try {
  for (const promptType of ['prose', 'code']) {
    const started = decodeBenchManager.start({
      sparkId: `tp4-${promptType}`, lanIp: benchHost, host: benchHost, port: benchPort,
      modelId, concurrencies: [1, 2, 3, 4], maxTokens: 400, promptType,
    });
    const job = await wait(decodeBenchManager, started.benchId, promptType);
    result.jobs[promptType] = job;
    save();
    if (job.status !== 'completed' || job.results.some(r => r.error)) throw Error(`${promptType} failed: ${job.error || JSON.stringify(job.results.map(r => r.error))}`);
  }
  const started = prefillBenchManager.start({
    sparkId: 'tp4-prefill', lanIp: benchHost, host: benchHost, port: benchPort,
    modelId, contextSizes: [8192, 16384, 32768, 65536, 131072, 262144],
  });
  const job = await wait(prefillBenchManager, started.benchId, 'prefill');
  result.jobs.prefill = job;
  save();
  if (job.status !== 'completed' || job.results.some(r => r.error)) throw Error(`prefill failed: ${job.error || JSON.stringify(job.results.map(r => r.error))}`);
  result.completedAt = new Date().toISOString();
  save();
} catch (error) {
  result.error = String(error);
  result.completedAt = new Date().toISOString();
  save();
  console.error(error);
  process.exitCode = 1;
} finally {
  await closeLlmStreamAgent();
}
