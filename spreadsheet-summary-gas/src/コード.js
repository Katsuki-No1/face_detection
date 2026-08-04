const SUMMARY_CONFIG = Object.freeze({
  summarySheetName: '全体の内容',
  monthlySheetPattern: /^\s*(?:(\d{4})年)?(\d{1,2})月の進捗\s*$/,
  titleRow: 1,
  descriptionRow: 2,
  groupHeaderRow: 3,
  headerRow: 4,
  dataStartRow: 5,
  baseColumnCount: 8,
  monthlyMetricColumnCount: 3,
  colors: Object.freeze({
    title: '#17324D',
    titleText: '#FFFFFF',
    description: '#DDEBF7',
    baseGroup: '#365F91',
    monthGroups: Object.freeze(['#2F75B5', '#548235']),
    header: '#D9EAF7',
    manual: '#FFF2CC',
    calculated: '#E2F0D9',
    monthData: Object.freeze(['#EAF3F8', '#EDF4E8']),
    border: '#A6B1BB',
    text: '#1F2933'
  })
});

/**
 * スプレッドシートを開いたときに、メニューを追加してまとめを更新する。
 */
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('集計')
    .addItem('まとめを更新', 'updateSummary')
    .addToUi();

  try {
    updateSummary_({ notify: false });
  } catch (error) {
    console.error(error && error.stack ? error.stack : error);
    SpreadsheetApp.getActiveSpreadsheet().toast(
      'まとめの自動更新に失敗しました。集計メニューから再実行してください。',
      '集計エラー',
      8
    );
  }
}

/**
 * カスタムメニューから実行する公開関数。
 */
function updateSummary() {
  updateSummary_({ notify: true });
}

/**
 * 月別シートを集計し、「全体の内容」を再描画する。
 * 手入力の加入日・印象欄は名前をキーにして引き継ぐ。
 *
 * @param {{notify: boolean}} options 実行完了をトースト表示するか。
 */
function updateSummary_(options) {
  const lock = LockService.getDocumentLock();
  if (!lock.tryLock(5000)) {
    throw new Error('別の集計処理が実行中です。少し待ってから再実行してください。');
  }

  try {
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const monthlySheets = getMonthlySheets_(spreadsheet);
    if (monthlySheets.length === 0) {
      throw new Error('「○月の進捗」という名前の月別シートが見つかりません。');
    }

    const parsedMonths = monthlySheets.map(function(month) {
      return {
        key: month.key,
        label: month.label,
        title: month.sheet.getName(),
        workers: parseMonthlyValues_(month.sheet.getDataRange().getValues())
      };
    });
    const model = buildSummaryModel_(parsedMonths);
    if (model.workers.length === 0) {
      throw new Error('月別シートからメンバー情報を読み取れませんでした。');
    }

    let summarySheet = spreadsheet.getSheetByName(SUMMARY_CONFIG.summarySheetName);
    if (!summarySheet) {
      summarySheet = spreadsheet.insertSheet(SUMMARY_CONFIG.summarySheetName, 0);
    }

    const manualEntries = readManualEntries_(summarySheet);
    renderSummary_(summarySheet, model, manualEntries, spreadsheet.getSpreadsheetTimeZone());

    if (options && options.notify) {
      spreadsheet.toast('「全体の内容」を更新しました。', '集計完了', 4);
    }
  } finally {
    lock.releaseLock();
  }
}

/**
 * シートの並び順を維持したまま月別シートを取得する。
 *
 * @param {GoogleAppsScript.Spreadsheet.Spreadsheet} spreadsheet
 * @return {Array<{key: string, label: string, sheet: GoogleAppsScript.Spreadsheet.Sheet}>}
 */
function getMonthlySheets_(spreadsheet) {
  return spreadsheet.getSheets().reduce(function(months, sheet) {
    const match = sheet.getName().match(SUMMARY_CONFIG.monthlySheetPattern);
    if (!match) {
      return months;
    }

    const year = match[1];
    const month = Number(match[2]);
    if (month < 1 || month > 12) {
      return months;
    }

    months.push({
      key: String(sheet.getSheetId()),
      label: year ? year + '年' + month + '月' : month + '月',
      sheet: sheet
    });
    return months;
  }, []);
}

/**
 * 1枚の月別シートからメンバー別の集計値を作る。
 *
 * @param {Array<Array<*>>} values 月別シートの値。
 * @return {Array<{key: string, name: string, headCount: number, frameCount: number, completedTaskCount: number, completedTaskMax: number}>}
 */
function parseMonthlyValues_(values) {
  const workers = [];
  const workerIndex = {};

  for (let row = 0; row < values.length; row += 1) {
    if (!isWorkerBlockStart_(values, row)) {
      continue;
    }

    const name = normalizeText_(values[row][0]);
    const key = normalizeNameKey_(name);
    const taskHeaders = values[row + 1] || [];
    const completionRow = findLabeledRow_(values, row + 2, row + 9, '修正完了');
    const headRow = findLabeledRow_(values, row + 2, row + 9, 'HEAD数');
    const frameRow = findLabeledRow_(values, row + 2, row + 9, 'frame数');
    let completedTaskCount = 0;
    let completedTaskMax = 0;
    let headCount = 0;
    let frameCount = 0;

    for (let column = 1; column < taskHeaders.length; column += 1) {
      const taskNumber = extractTaskNumber_(taskHeaders[column]);
      if (taskNumber === null) {
        continue;
      }

      if (completionRow && isCompletionMark_(completionRow[column])) {
        completedTaskCount += 1;
        completedTaskMax = Math.max(completedTaskMax, taskNumber);
      }
      if (headRow) {
        headCount += toNumberOrZero_(headRow[column]);
      }
      if (frameRow) {
        frameCount += toNumberOrZero_(frameRow[column]);
      }
    }

    if (Object.prototype.hasOwnProperty.call(workerIndex, key)) {
      const existing = workers[workerIndex[key]];
      existing.completedTaskCount += completedTaskCount;
      existing.completedTaskMax = Math.max(existing.completedTaskMax, completedTaskMax);
      existing.headCount += headCount;
      existing.frameCount += frameCount;
    } else {
      workerIndex[key] = workers.length;
      workers.push({
        key: key,
        name: name,
        headCount: headCount,
        frameCount: frameCount,
        completedTaskCount: completedTaskCount,
        completedTaskMax: completedTaskMax
      });
    }

    row += 7;
  }

  return workers;
}

/**
 * 複数月のデータを、まとめシート用のモデルへ統合する。
 *
 * @param {Array<{key: string, label: string, title: string, workers: Array}>} months
 * @return {{months: Array, workers: Array}}
 */
function buildSummaryModel_(months) {
  const workers = [];
  const workerIndex = {};

  months.forEach(function(month) {
    month.workers.forEach(function(monthWorker) {
      if (!Object.prototype.hasOwnProperty.call(workerIndex, monthWorker.key)) {
        workerIndex[monthWorker.key] = workers.length;
        workers.push({
          key: monthWorker.key,
          name: monthWorker.name,
          currentTask: 1,
          totalHeadCount: 0,
          totalFrameCount: 0,
          completedTaskMax: 0,
          monthly: {}
        });
      }

      const worker = workers[workerIndex[monthWorker.key]];
      worker.monthly[month.key] = {
        headCount: monthWorker.headCount,
        frameCount: monthWorker.frameCount,
        completedTaskCount: monthWorker.completedTaskCount,
        completedTaskMax: monthWorker.completedTaskMax
      };
      worker.totalHeadCount += monthWorker.headCount;
      worker.totalFrameCount += monthWorker.frameCount;
      worker.completedTaskMax = Math.max(
        worker.completedTaskMax,
        monthWorker.completedTaskMax
      );
      worker.currentTask = worker.completedTaskMax + 1;
    });
  });

  return {
    months: months.map(function(month) {
      return { key: month.key, label: month.label, title: month.title };
    }),
    workers: workers
  };
}

/**
 * 既存まとめシートから手入力項目を取得する。
 *
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet
 * @return {Object<string, {joinDate: *, revisionImpression: *, communicationImpression: *}>}
 */
function readManualEntries_(sheet) {
  const values = sheet.getDataRange().getValues();
  const entries = {};
  let headerRowIndex = -1;
  let headers = [];

  for (let row = 0; row < Math.min(values.length, 10); row += 1) {
    if (values[row].some(function(value) { return normalizeText_(value) === '名前'; })) {
      headerRowIndex = row;
      headers = values[row].map(normalizeText_);
      break;
    }
  }
  if (headerRowIndex < 0) {
    return entries;
  }

  const nameColumn = headers.indexOf('名前');
  const joinDateColumn = headers.indexOf('加入日');
  let revisionColumn = headers.indexOf('修正の量の印象');
  if (revisionColumn < 0) {
    revisionColumn = headers.indexOf('修正量の印象');
  }
  const communicationColumn = headers.indexOf('連絡の印象');

  for (let row = headerRowIndex + 1; row < values.length; row += 1) {
    const name = nameColumn >= 0 ? normalizeText_(values[row][nameColumn]) : '';
    if (!name) {
      continue;
    }
    entries[normalizeNameKey_(name)] = {
      joinDate: joinDateColumn >= 0 ? values[row][joinDateColumn] : '',
      revisionImpression: revisionColumn >= 0 ? values[row][revisionColumn] : '',
      communicationImpression: communicationColumn >= 0 ? values[row][communicationColumn] : ''
    };
  }

  return entries;
}

/**
 * まとめシートを再描画する。
 *
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet
 * @param {{months: Array, workers: Array}} model
 * @param {Object} manualEntries
 * @param {string} timeZone
 */
function renderSummary_(sheet, model, manualEntries, timeZone) {
  const config = SUMMARY_CONFIG;
  const totalColumns = config.baseColumnCount +
    model.months.length * config.monthlyMetricColumnCount;
  const requiredRows = config.dataStartRow + model.workers.length - 1;
  ensureSheetSize_(sheet, requiredRows, totalColumns);

  const filter = sheet.getFilter();
  if (filter) {
    filter.remove();
  }
  sheet.getBandings().forEach(function(banding) { banding.remove(); });
  sheet.getDataRange().breakApart();
  sheet.setFrozenColumns(0);
  sheet.clear();
  sheet.setConditionalFormatRules([]);

  const titleRange = sheet.getRange(config.titleRow, 1, 1, totalColumns);
  titleRange.merge();
  titleRange.setValue('ワーカーさんのタスク状況｜まとめ');

  const updatedAt = Utilities.formatDate(new Date(), timeZone || 'Asia/Tokyo', 'yyyy/MM/dd HH:mm');
  const descriptionRange = sheet.getRange(config.descriptionRow, 1, 1, totalColumns);
  descriptionRange.merge();
  descriptionRange.setValue(
    '最終更新: ' + updatedAt + '　｜　黄色の列は手入力、その他は月別シートから自動集計'
  );

  const baseGroupRange = sheet.getRange(config.groupHeaderRow, 1, 1, config.baseColumnCount);
  baseGroupRange.merge();
  baseGroupRange.setValue('基本情報');

  model.months.forEach(function(month, index) {
    const firstColumn = config.baseColumnCount +
      index * config.monthlyMetricColumnCount + 1;
    const groupRange = sheet.getRange(
      config.groupHeaderRow,
      firstColumn,
      1,
      config.monthlyMetricColumnCount
    );
    groupRange.merge();
    groupRange.setValue(month.label);
  });

  const headers = [
    'No.',
    '名前',
    '加入日',
    '現在のtask',
    'HEAD数の合計',
    'frame数の合計',
    '修正の量の印象',
    '連絡の印象'
  ];
  model.months.forEach(function() {
    headers.push('HEAD数', 'task数', 'frame数');
  });
  sheet.getRange(config.headerRow, 1, 1, totalColumns).setValues([headers]);

  const data = model.workers.map(function(worker, index) {
    const manual = manualEntries[worker.key] || {};
    const row = [
      index + 1,
      worker.name,
      manual.joinDate || '',
      worker.currentTask,
      worker.totalHeadCount,
      worker.totalFrameCount,
      manual.revisionImpression || '',
      manual.communicationImpression || ''
    ];
    model.months.forEach(function(month) {
      const metrics = worker.monthly[month.key] || {
        headCount: 0,
        frameCount: 0,
        completedTaskCount: 0,
        completedTaskMax: 0
      };
      row.push(metrics.headCount, metrics.completedTaskCount, metrics.frameCount);
    });
    return row;
  });
  sheet.getRange(config.dataStartRow, 1, data.length, totalColumns).setValues(data);

  applySummaryFormatting_(sheet, model, totalColumns, data.length);
}

/**
 * まとめシートの視認性を高める書式を適用する。
 */
function applySummaryFormatting_(sheet, model, totalColumns, dataRowCount) {
  const config = SUMMARY_CONFIG;
  const colors = config.colors;
  const fullRange = sheet.getRange(1, 1, config.dataStartRow + dataRowCount - 1, totalColumns);
  fullRange.setFontFamily('Arial').setFontColor(colors.text).setVerticalAlignment('middle');

  sheet.getRange(config.titleRow, 1, 1, totalColumns)
    .setBackground(colors.title)
    .setFontColor(colors.titleText)
    .setFontSize(16)
    .setFontWeight('bold')
    .setHorizontalAlignment('left');
  sheet.setRowHeight(config.titleRow, 42);

  sheet.getRange(config.descriptionRow, 1, 1, totalColumns)
    .setBackground(colors.description)
    .setFontColor(colors.title)
    .setFontSize(10)
    .setHorizontalAlignment('left');
  sheet.setRowHeight(config.descriptionRow, 28);

  sheet.getRange(config.groupHeaderRow, 1, 1, config.baseColumnCount)
    .setBackground(colors.baseGroup)
    .setFontColor(colors.titleText)
    .setFontWeight('bold')
    .setHorizontalAlignment('center');
  model.months.forEach(function(month, index) {
    const firstColumn = config.baseColumnCount +
      index * config.monthlyMetricColumnCount + 1;
    sheet.getRange(
      config.groupHeaderRow,
      firstColumn,
      1,
      config.monthlyMetricColumnCount
    )
      .setBackground(colors.monthGroups[index % colors.monthGroups.length])
      .setFontColor(colors.titleText)
      .setFontWeight('bold')
      .setHorizontalAlignment('center');
  });
  sheet.setRowHeight(config.groupHeaderRow, 30);

  const headerRange = sheet.getRange(config.headerRow, 1, 1, totalColumns);
  headerRange
    .setBackground(colors.header)
    .setFontWeight('bold')
    .setHorizontalAlignment('center')
    .setWrap(true);
  sheet.setRowHeight(config.headerRow, 38);

  const dataRange = sheet.getRange(config.dataStartRow, 1, dataRowCount, totalColumns);
  dataRange.setBackground('#FFFFFF').setWrap(true);
  sheet.getRange(config.dataStartRow, 1, dataRowCount, 1).setHorizontalAlignment('center');
  sheet.getRange(config.dataStartRow, 2, dataRowCount, 1).setFontWeight('bold');
  sheet.getRange(config.dataStartRow, 3, dataRowCount, 1)
    .setBackground(colors.manual)
    .setNumberFormat('yyyy/mm/dd')
    .setHorizontalAlignment('center');
  sheet.getRange(config.dataStartRow, 4, dataRowCount, 3)
    .setBackground(colors.calculated)
    .setNumberFormat('0')
    .setHorizontalAlignment('center')
    .setFontWeight('bold');
  sheet.getRange(config.dataStartRow, 7, dataRowCount, 2).setBackground(colors.manual);

  model.months.forEach(function(month, index) {
    const firstColumn = config.baseColumnCount +
      index * config.monthlyMetricColumnCount + 1;
    sheet.getRange(
      config.dataStartRow,
      firstColumn,
      dataRowCount,
      config.monthlyMetricColumnCount
    )
      .setBackground(colors.monthData[index % colors.monthData.length])
      .setNumberFormat('0')
      .setHorizontalAlignment('center');
  });

  sheet.getRange(config.groupHeaderRow, 1, dataRowCount + 2, totalColumns)
    .setBorder(true, true, true, true, true, true, colors.border, SpreadsheetApp.BorderStyle.SOLID);

  sheet.setColumnWidth(1, 52);
  sheet.setColumnWidth(2, 145);
  sheet.setColumnWidth(3, 105);
  sheet.setColumnWidth(4, 105);
  sheet.setColumnWidth(5, 115);
  sheet.setColumnWidth(6, 120);
  sheet.setColumnWidth(7, 190);
  sheet.setColumnWidth(8, 190);
  for (let column = 9; column <= totalColumns; column += 1) {
    sheet.setColumnWidth(column, 88);
  }
  for (let row = config.dataStartRow; row < config.dataStartRow + dataRowCount; row += 1) {
    sheet.setRowHeight(row, 34);
  }

  sheet.getRange(config.headerRow, 3).setNote('手入力欄です。自動更新時も値を保持します。');
  sheet.getRange(config.headerRow, 7, 1, 2).setNote('手入力欄です。自動更新時も値を保持します。');
  sheet.getRange(config.headerRow, 4).setNote('完了task番号の最大値 + 1 を自動表示します。');
  sheet.getRange(config.headerRow, 5).setNote('すべての月別シートのHEAD数を合計して自動表示します。');
  sheet.getRange(config.headerRow, 6).setNote('すべての月別シートのframe数を合計して自動表示します。');

  sheet.getRange(config.headerRow, 1, dataRowCount + 1, totalColumns).createFilter();
  sheet.setFrozenRows(config.headerRow);
  sheet.setHiddenGridlines(true);
  sheet.setTabColor(colors.title);
}

function ensureSheetSize_(sheet, requiredRows, requiredColumns) {
  if (sheet.getMaxRows() < requiredRows) {
    sheet.insertRowsAfter(sheet.getMaxRows(), requiredRows - sheet.getMaxRows());
  }
  if (sheet.getMaxColumns() < requiredColumns) {
    sheet.insertColumnsAfter(sheet.getMaxColumns(), requiredColumns - sheet.getMaxColumns());
  }
}

function isWorkerBlockStart_(values, row) {
  if (!normalizeText_(values[row] && values[row][0]) || row + 7 >= values.length) {
    return false;
  }
  return normalizeText_(values[row + 2] && values[row + 2][0]) === 'task作成' &&
    normalizeText_(values[row + 6] && values[row + 6][0]) === '修正完了' &&
    normalizeText_(values[row + 7] && values[row + 7][0]) === 'HEAD数';
}

function findLabeledRow_(values, startRow, endRow, label) {
  const lastRow = Math.min(endRow, values.length - 1);
  for (let row = startRow; row <= lastRow; row += 1) {
    if (normalizeText_(values[row] && values[row][0]) === label) {
      return values[row];
    }
  }
  return null;
}

function extractTaskNumber_(value) {
  const normalized = normalizeDigits_(normalizeText_(value));
  const match = normalized.match(/(\d+)\s*$/);
  return match ? Number(match[1]) : null;
}

function isCompletionMark_(value) {
  return /^[〇○]$/.test(normalizeText_(value));
}

function toNumberOrZero_(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : 0;
  }
  const normalized = normalizeDigits_(normalizeText_(value)).replace(/,/g, '');
  if (!normalized) {
    return 0;
  }
  const number = Number(normalized);
  return Number.isFinite(number) ? number : 0;
}

function normalizeNameKey_(value) {
  return normalizeText_(value).replace(/[\s　]+/g, '').toLowerCase();
}

function normalizeText_(value) {
  if (value === null || typeof value === 'undefined') {
    return '';
  }
  return String(value).trim();
}

function normalizeDigits_(value) {
  return String(value).replace(/[０-９]/g, function(character) {
    return String.fromCharCode(character.charCodeAt(0) - 0xFEE0);
  });
}
