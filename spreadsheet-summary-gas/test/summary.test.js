const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');

function loadPureFunctions() {
  const source = fs.readFileSync(
    path.join(__dirname, '..', 'src', 'コード.js'),
    'utf8'
  );
  const context = vm.createContext({ console });
  vm.runInContext(
    source + '\nthis.__testExports = { parseMonthlyValues_, buildSummaryModel_ };',
    context
  );
  return context.__testExports;
}

const { parseMonthlyValues_, buildSummaryModel_ } = loadPureFunctions();

test('修正完了の〇だけをtask完了として扱い、HEAD数とframe数を合計する', () => {
  const values = [
    ['yukaさん'],
    ['', 'task5', 'task6', 'task7', 'task8'],
    ['task作成', '〇', '〇', '〇', '〇'],
    ['作業完了', '〇', '〇', '', ''],
    ['確認', '〇', '〇', '', ''],
    ['修正連絡', '〇', '〇', '', ''],
    ['修正完了', '〇', '〇', '', ''],
    ['HEAD数', 480, 433, '', ''],
    ['frame数', 299, 300, '', ''],
    [],
    ['新規さん'],
    ['', 'test'],
    ['task作成', '〇'],
    ['作業完了', '〇'],
    ['確認', '〇'],
    ['修正連絡', '〇'],
    ['修正完了', '〇'],
    ['HEAD数', 999]
  ];

  const workers = parseMonthlyValues_(values);

  assert.deepEqual(JSON.parse(JSON.stringify(workers)), [
    {
      key: 'yukaさん',
      name: 'yukaさん',
      headCount: 913,
      frameCount: 599,
      completedTaskCount: 2,
      completedTaskMax: 6
    },
    {
      key: '新規さん',
      name: '新規さん',
      headCount: 0,
      frameCount: 0,
      completedTaskCount: 0,
      completedTaskMax: 0
    }
  ]);
});

test('tas1をtask1として扱い、全期間の最大完了task + 1を現在taskにする', () => {
  const juneWorkers = parseMonthlyValues_([
    ['yukaさん'],
    ['', 'tas1', 'task2', 'task3', 'task4'],
    ['task作成', '〇', '〇', '〇', '〇'],
    ['作業完了', '〇', '〇', '〇', '〇'],
    ['確認', '〇', '〇', '〇', '〇'],
    ['修正連絡', '〇', '〇', '〇', '〇'],
    ['修正完了', '〇', '〇', '〇', '〇'],
    ['HEAD数', '', '', '', ''],
    ['frame数', 300, 300, 300, 300]
  ]);
  const julyWorkers = parseMonthlyValues_([
    ['yukaさん'],
    ['', 'task5', 'task6', 'task7'],
    ['task作成', '〇', '〇', '〇'],
    ['作業完了', '〇', '〇', ''],
    ['確認', '〇', '〇', ''],
    ['修正連絡', '〇', '〇', ''],
    ['修正完了', '〇', '〇', ''],
    ['HEAD数', 480, 433, ''],
    ['frame数', 299, 300, '']
  ]);

  const model = buildSummaryModel_([
    { key: 'june', label: '6月', title: '6月の進捗', workers: juneWorkers },
    { key: 'july', label: '7月', title: '7月の進捗', workers: julyWorkers }
  ]);

  assert.equal(model.workers[0].monthly.june.completedTaskMax, 4);
  assert.equal(model.workers[0].monthly.june.completedTaskCount, 4);
  assert.equal(model.workers[0].monthly.june.headCount, 0);
  assert.equal(model.workers[0].monthly.july.completedTaskMax, 6);
  assert.equal(model.workers[0].monthly.july.completedTaskCount, 2);
  assert.equal(model.workers[0].totalHeadCount, 913);
  assert.equal(model.workers[0].monthly.june.frameCount, 1200);
  assert.equal(model.workers[0].monthly.july.frameCount, 599);
  assert.equal(model.workers[0].totalFrameCount, 1799);
  assert.equal(model.workers[0].currentTask, 7);
});

test('各月のHEAD数とframe数をメンバーごとに合計する', () => {
  const model = buildSummaryModel_([
    {
      key: 'june',
      label: '6月',
      title: '6月の進捗',
      workers: [{
        key: 'worker',
        name: 'worker',
        headCount: 120,
        frameCount: 40,
        completedTaskCount: 2,
        completedTaskMax: 2
      }]
    },
    {
      key: 'july',
      label: '7月',
      title: '7月の進捗',
      workers: [{
        key: 'worker',
        name: 'worker',
        headCount: 380,
        frameCount: 60,
        completedTaskCount: 2,
        completedTaskMax: 4
      }]
    }
  ]);

  assert.equal(model.workers[0].totalHeadCount, 500);
  assert.equal(model.workers[0].totalFrameCount, 100);
});
