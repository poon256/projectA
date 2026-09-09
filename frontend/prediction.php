<?php
session_start();

/*
| Prediction Dashboard
| Regression:
|   model/Regression_Linear.py
|
| Classification:
|   model/Classification_RandomForest.py
|
| Database:
|   projecta
*/

$result = null;
$error = null;

$province = $_POST['province'] ?? 'สมุทรสงคราม';
$year     = isset($_POST['year']) ? (int)$_POST['year'] : 2569;
$month    = isset($_POST['month']) ? (int)$_POST['month'] : 1;

$monthNames = [
    1  => 'มกราคม',
    2  => 'กุมภาพันธ์',
    3  => 'มีนาคม',
    4  => 'เมษายน',
    5  => 'พฤษภาคม',
    6  => 'มิถุนายน',
    7  => 'กรกฎาคม',
    8  => 'สิงหาคม',
    9  => 'กันยายน',
    10 => 'ตุลาคม',
    11 => 'พฤศจิกายน',
    12 => 'ธันวาคม'
];

/*
| ฟังก์ชันเรียก Python
*/

function runPython($script, $province, $year, $month)
{
    $python = 'python';

    $province64 = base64_encode($province);

    $command =
        escapeshellarg($python)
        . " "
        . escapeshellarg($script)
        . " "
        . escapeshellarg($province64)
        . " "
        . escapeshellarg($year)
        . " "
        . escapeshellarg($month)
        . " --b64 2>&1";

    $output = shell_exec($command);

    if ($output === null || trim($output) === '') {
        return [
            'status' => 'error',
            'message' => 'ไม่สามารถเรียก Python ได้'
        ];
    }

    /*
     * Python อาจมีข้อความ warning ก่อน JSON
     * จึงหา JSON object จาก output
     */
    $lines = preg_split('/\R/', trim($output));

    for ($i = count($lines) - 1; $i >= 0; $i--) {

        $line = trim($lines[$i]);

        if ($line === '') {
            continue;
        }

        $json = json_decode($line, true);

        if (json_last_error() === JSON_ERROR_NONE && is_array($json)) {
            return $json;
        }
    }

    return [
        'status' => 'error',
        'message' => 'Python ไม่ได้ส่งข้อมูลกลับมาเป็น JSON',
        'raw_output' => $output
    ];
}


/*
| Prediction
*/

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['predict'])) {

    if ($month < 1 || $month > 12) {
        $error = 'เดือนต้องอยู่ระหว่าง 1-12';
    } elseif ($year < 2562 || $year > 2570) {
        $error = 'ปีต้องอยู่ระหว่าง 2562-2570';
    } else {

        /*
        | 1. Regression
        */

        $regressionScript = realpath(
            __DIR__ . '/../model/Regression_Linear.py'
        );

        if (!$regressionScript || !file_exists($regressionScript)) {

            $error = 'ไม่พบ Regression_Linear.py';

        } else {

            $regression = runPython(
                $regressionScript,
                $province,
                $year,
                $month
            );

            /*
            | 2. Classification
            */

            $classificationScript = realpath(
                __DIR__ . '/../model/Classification_RandomForest.py'
            );

            if (!$classificationScript || !file_exists($classificationScript)) {

                $classification = [
                    'status' => 'error',
                    'message' => 'ไม่พบ Classification_RandomForest.py'
                ];

            } else {

                $classification = runPython(
                    $classificationScript,
                    $province,
                    $year,
                    $month
                );
            }


            /*
            | ตรวจสอบ Regression
            */

            if (($regression['status'] ?? '') !== 'success') {

                $error =
                    'Regression Error: '
                    . ($regression['message'] ?? 'ไม่ทราบสาเหตุ');

            } else {

                /*
                | รวมผล
                */

                $result = [

                    'province' =>
                        $regression['province']
                        ?? $province,

                    'year' =>
                        $regression['year']
                        ?? $year,

                    'month' =>
                        $regression['month']
                        ?? $month,

                    'ton' =>
                        $regression['ton']
                        ?? 0,

                    'actual_ton' =>
                        $regression['actual_ton']
                        ?? null,

                    'sst' =>
                        $regression['sst']
                        ?? null,

                    'chlor_a' =>
                        $regression['chlor_a']
                        ?? null,

                    'prediction_mode' =>
                        $regression['prediction_mode']
                        ?? '',

                    'data_source' =>
                        $regression['data_source']
                        ?? '',

                    'classification' =>
                        $classification
                ];
            }
        }
    }
}


/*
| Cluster จาก dataset_ml
*/

$cluster = null;

$conn = @new mysqli(
    'localhost',
    'root',
    '',
    'projecta'
);

if (!$conn->connect_error && $result) {

    $stationMap = [
        'เพชรบุรี'      => 1,
        'สมุทรสงคราม'  => 2,
        'สมุทรสาคร'    => 3,
        'ชลบุรี'       => 4,
        'สมุทรปราการ'  => 5
    ];

    $stationId = $stationMap[$province] ?? null;

    if ($stationId !== null) {

        $stmt = $conn->prepare("
            SELECT cluster
            FROM dataset_ml
            WHERE station_id = ?
              AND year = ?
              AND month = ?
            ORDER BY id DESC
            LIMIT 1
        ");

        if ($stmt) {

            $stmt->bind_param(
                "iii",
                $stationId,
                $year,
                $month
            );

            $stmt->execute();

            $queryResult = $stmt->get_result();

            if ($row = $queryResult->fetch_assoc()) {
                $cluster = $row['cluster'];
            }

            $stmt->close();
        }
    }

    $conn->close();
}


/*
| Classification Data
*/

$classification = $result['classification'] ?? [];

$level = $classification['level'] ?? null;

if (!$level) {
    $level = $classification['density_level'] ?? null;
}

$accuracy = $classification['accuracy'] ?? null;

$precision = $classification['precision'] ?? null;
$recall    = $classification['recall'] ?? null;
$f1        = $classification['f1'] ?? null;

$probability = $classification['probability'] ?? [];

$description =
    $classification['description']
    ?? '';


$badgeClass = 'bg-secondary';

if ($level === 'LOW') {
    $badgeClass = 'bg-success';
}

if ($level === 'MEDIUM') {
    $badgeClass = 'bg-warning text-dark';
}

if ($level === 'HIGH') {
    $badgeClass = 'bg-danger';
}


$spawningMonths = [
    1, 2, 3, 4
];

$isSpawning =
    in_array($month, $spawningMonths);


/*
| AI Summary
*/

$summary = '';

if ($result) {

    if ($level === 'HIGH') {

        $summary =
            "พื้นที่ {$province} ในเดือน {$monthNames[$month]} "
            . "มีแนวโน้มความหนาแน่นของปลาทูอยู่ในระดับสูง "
            . "จากผลการจำแนกของ Random Forest";

    } elseif ($level === 'MEDIUM') {

        $summary =
            "พื้นที่ {$province} ในเดือน {$monthNames[$month]} "
            . "มีแนวโน้มความหนาแน่นของปลาทูอยู่ในระดับปานกลาง";

    } elseif ($level === 'LOW') {

        $summary =
            "พื้นที่ {$province} ในเดือน {$monthNames[$month]} "
            . "มีแนวโน้มความหนาแน่นของปลาทูอยู่ในระดับต่ำ";

    } else {

        $summary =
            "ระบบสามารถคาดการณ์จำนวนปลาทูได้ "
            . "แต่ไม่สามารถจำแนกระดับความหนาแน่นได้";
    }
}

?>

<!DOCTYPE html>
<html lang="th">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Prediction Dashboard</title>

<link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
>

<link
    rel="stylesheet"
    href="../css/menu.css"
>

<link
    rel="stylesheet"
    href="../css/data.css"
>

<link
    rel="stylesheet"
    href="../css/footer.css"
>

<style>

.result-card {
    min-height: 170px;
}

.metric-number {
    font-size: 32px;
    font-weight: 700;
}

.environment-card {
    border-left: 4px solid #0d6efd;
}

.probability-bar {
    height: 24px;
}

.heatmap-box {
    border-radius: 10px;
    padding: 25px;
    background: #f8f9fa;
}

</style>

</head>

<body>

<?php include 'menu.php'; ?>


<div class="container page-container">

    <!--HEADER-->

    <div class="page-header mb-4">

        <h2>
            Prediction Dashboard
        </h2>

        <p class="text-muted">
            Integrated AI Prediction System
        </p>

    </div>


    <!--ERROR-->

    <?php if ($error): ?>

        <div class="alert alert-danger">

            <strong>Prediction Error</strong><br>

            <?= htmlspecialchars($error) ?>

        </div>

    <?php endif; ?>


    <!--INPUT-->

    <div class="card shadow-sm border-0 mb-4">

        <div class="card-header">
            <strong>Prediction Parameters</strong>
        </div>

        <div class="card-body">

            <form method="POST">

                <div class="row">

                    <!-- MONTH -->

                    <div class="col-md-4 mb-3">

                        <label class="form-label">
                            Month
                        </label>

                        <select
                            name="month"
                            class="form-select"
                            required
                        >

                            <?php foreach ($monthNames as $number => $name): ?>

                                <option
                                    value="<?= $number ?>"
                                    <?= $month == $number ? 'selected' : '' ?>
                                >

                                    <?= $name ?>

                                </option>

                            <?php endforeach; ?>

                        </select>

                    </div>


                    <!-- YEAR -->

                    <div class="col-md-4 mb-3">

                        <label class="form-label">
                            Year
                        </label>

                        <input
                            type="number"
                            name="year"
                            class="form-control"
                            min="2562"
                            max="2570"
                            value="<?= htmlspecialchars($year) ?>"
                            required
                        >

                    </div>


                    <!-- PROVINCE -->

                    <div class="col-md-4 mb-3">

                        <label class="form-label">
                            Province
                        </label>

                        <select
                            name="province"
                            class="form-select"
                            required
                        >

                            <?php
                            $provinces = [
                                'สมุทรปราการ',
                                'สมุทรสาคร',
                                'สมุทรสงคราม',
                                'เพชรบุรี',
                                'ชลบุรี'
                            ];
                            ?>

                            <?php foreach ($provinces as $p): ?>

                                <option
                                    value="<?= htmlspecialchars($p) ?>"
                                    <?= $province === $p ? 'selected' : '' ?>
                                >

                                    <?= htmlspecialchars($p) ?>

                                </option>

                            <?php endforeach; ?>

                        </select>

                    </div>

                </div>


                <button
                    type="submit"
                    name="predict"
                    class="btn btn-primary px-4"
                >

                    Predict

                </button>

            </form>

        </div>

    </div>


    <?php if ($result): ?>


    <!-- SPAWNING-->

    <?php if ($isSpawning): ?>

        <div class="alert alert-warning mb-4">

            ⚠️ เดือนที่เลือกอยู่ในช่วงที่ระบบกำหนดให้เป็นช่วงเฝ้าระวัง
            ฤดูวางไข่ของปลาทู

        </div>

    <?php endif; ?>


    <!--MAIN RESULT-->

    <div class="row mb-4">

        <!-- REGRESSION -->

        <div class="col-md-4 mb-3">

            <div class="card shadow-sm result-card">

                <div class="card-body">

                    <h5>
                        Regression
                    </h5>

                    <div class="metric-number text-primary">

                        <?= number_format(
                            (float)$result['ton'],
                            2
                        ) ?>

                    </div>

                    <small class="text-muted">
                        Ton
                    </small>

                </div>

            </div>

        </div>


        <!-- CLASSIFICATION -->

        <div class="col-md-4 mb-3">

            <div class="card shadow-sm result-card">

                <div class="card-body">

                    <h5>
                        Classification
                    </h5>

                    <?php if ($level): ?>

                        <span
                            class="badge <?= $badgeClass ?> fs-5 px-4 py-2"
                        >

                            <?= htmlspecialchars($level) ?>

                        </span>

                        <div class="mt-2 text-muted">
                            Density Level
                        </div>

                    <?php else: ?>

                        <span class="badge bg-secondary fs-6">
                            N/A
                        </span>

                    <?php endif; ?>

                </div>

            </div>

        </div>


        <!-- CLUSTER -->

        <div class="col-md-4 mb-3">

            <div class="card shadow-sm result-card">

                <div class="card-body">

                    <h5>
                        Cluster
                    </h5>

                    <div class="metric-number">

                        <?= $cluster !== null
                            ? htmlspecialchars($cluster)
                            : 'N/A'
                        ?>

                    </div>

                    <small class="text-muted">
                        Environmental Cluster
                    </small>

                </div>

            </div>

        </div>

    </div>


    <!--ENVIRONMENT-->

    <div class="card shadow-sm border-0 mb-4">

        <div class="card-header">

            <strong>
                Environmental Conditions
            </strong>

        </div>

        <div class="card-body">

            <div class="row">

                <div class="col-md-4 mb-3">

                    <div class="card environment-card">

                        <div class="card-body">

                            <small class="text-muted">
                                Sea Surface Temperature
                            </small>

                            <h4>

                                <?= $result['sst'] !== null
                                    ? number_format(
                                        (float)$result['sst'],
                                        2
                                    ) . ' °C'
                                    : 'N/A'
                                ?>

                            </h4>

                            <small>
                                SST
                            </small>

                        </div>

                    </div>

                </div>


                <div class="col-md-4 mb-3">

                    <div class="card environment-card">

                        <div class="card-body">

                            <small class="text-muted">
                                Chlorophyll-a
                            </small>

                            <h4>

                                <?= $result['chlor_a'] !== null
                                    ? number_format(
                                        (float)$result['chlor_a'],
                                        2
                                    )
                                    : 'N/A'
                                ?>

                            </h4>

                            <small>
                                mg/m³
                            </small>

                        </div>

                    </div>

                </div>


                <div class="col-md-4 mb-3">

                    <div class="card environment-card">

                        <div class="card-body">

                            <small class="text-muted">
                                Prediction Mode
                            </small>

                            <h6 class="mt-2">

                                <?= htmlspecialchars(
                                    $result['prediction_mode']
                                ) ?>

                            </h6>

                        </div>

                    </div>

                </div>

            </div>

        </div>

    </div>


    <!--CLASSIFICATION PROBABILITY-->

    <?php if (!empty($probability)): ?>

        <div class="card shadow-sm border-0 mb-4">

            <div class="card-header">

                <strong>
                    Classification Probability
                </strong>

            </div>

            <div class="card-body">

                <?php foreach ($probability as $name => $value): ?>

                    <div class="mb-3">

                        <div class="d-flex justify-content-between">

                            <strong>
                                <?= htmlspecialchars($name) ?>
                            </strong>

                            <span>
                                <?= number_format(
                                    (float)$value,
                                    2
                                ) ?>%
                            </span>

                        </div>

                        <div class="progress probability-bar">

                            <div
                                class="progress-bar"
                                role="progressbar"
                                style="width: <?= min(
                                    100,
                                    max(0, (float)$value)
                                ) ?>%"
                            ></div>

                        </div>

                    </div>

                <?php endforeach; ?>

            </div>

        </div>

    <?php endif; ?>


    <!--MODEL PERFORMANCE-->

    <?php if (
        $accuracy !== null ||
        $precision !== null ||
        $recall !== null ||
        $f1 !== null
    ): ?>

        <div class="card shadow-sm border-0 mb-4">

            <div class="card-header">

                <strong>
                    Random Forest Model Performance
                </strong>

            </div>

            <div class="card-body">

                <div class="row text-center">

                    <?php if ($accuracy !== null): ?>

                        <div class="col-md-3 mb-3">

                            <h6>
                                Accuracy
                            </h6>

                            <h3>
                                <?= number_format(
                                    (float)$accuracy,
                                    2
                                ) ?>%
                            </h3>

                        </div>

                    <?php endif; ?>


                    <?php if ($precision !== null): ?>

                        <div class="col-md-3 mb-3">

                            <h6>
                                Precision
                            </h6>

                            <h3>
                                <?= number_format(
                                    (float)$precision,
                                    2
                                ) ?>%
                            </h3>

                        </div>

                    <?php endif; ?>


                    <?php if ($recall !== null): ?>

                        <div class="col-md-3 mb-3">

                            <h6>
                                Recall
                            </h6>

                            <h3>
                                <?= number_format(
                                    (float)$recall,
                                    2
                                ) ?>%
                            </h3>

                        </div>

                    <?php endif; ?>


                    <?php if ($f1 !== null): ?>

                        <div class="col-md-3 mb-3">

                            <h6>
                                F1 Score
                            </h6>

                            <h3>
                                <?= number_format(
                                    (float)$f1,
                                    2
                                ) ?>%
                            </h3>

                        </div>

                    <?php endif; ?>

                </div>

            </div>

        </div>

    <?php endif; ?>


    <!--FORECAST GRAPH-->

    <div class="card shadow-sm border-0 mb-4">

        <div class="card-header">

            <strong>
                Mackerel Catch Prediction
            </strong>

        </div>

        <div class="card-body">

            <img
                src="../model/output/regression_fish.png?<?= time() ?>"
                class="img-fluid"
                alt="Mackerel Catch Prediction"
            >

        </div>

    </div>


    <!--HEATMAP-->

    <div class="card shadow-sm border-0 mb-4">

        <div class="card-header">

            <strong>
                Environmental Heatmap
            </strong>

        </div>

        <div class="card-body">

            <div class="heatmap-box">

                <div class="row text-center">

                    <div class="col-md-4">

                        <strong>
                            Province
                        </strong>

                        <div class="mt-2">
                            <?= htmlspecialchars($province) ?>
                        </div>

                    </div>

                    <div class="col-md-4">

                        <strong>
                            Month
                        </strong>

                        <div class="mt-2">
                            <?= htmlspecialchars(
                                $monthNames[$month]
                            ) ?>
                        </div>

                    </div>

                    <div class="col-md-4">

                        <strong>
                            Cluster
                        </strong>

                        <div class="mt-2">

                            <?= $cluster !== null
                                ? htmlspecialchars($cluster)
                                : 'N/A'
                            ?>

                        </div>

                    </div>

                </div>

            </div>

        </div>

    </div>


    <!--AI SUMMARY -->
    <div class="card shadow-sm border-0 mb-4">

        <div class="card-header">

            <strong>
                AI Summary
            </strong>

        </div>

        <div class="card-body">

            <?= htmlspecialchars($summary) ?>

            <?php if ($result['ton'] !== null): ?>

                <br><br>

                ระบบคาดการณ์ปริมาณปลาทูประมาณ

                <strong class="text-primary">

                    <?= number_format(
                        (float)$result['ton'],
                        2
                    ) ?>

                    ตัน

                </strong>

                สำหรับ

                <strong>
                    <?= htmlspecialchars($province) ?>
                </strong>

                เดือน

                <strong>
                    <?= htmlspecialchars(
                        $monthNames[$month]
                    ) ?>
                </strong>

                ปี

                <strong>
                    <?= htmlspecialchars($year) ?>
                </strong>

            <?php endif; ?>

        </div>

    </div>


    <?php endif; ?>

</div>


<?php include 'footer.php'; ?>


<script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js">
</script>

</body>

</html>