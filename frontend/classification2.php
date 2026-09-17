<?php

session_start();

date_default_timezone_set("Asia/Bangkok");

$result = null;

$error = "";

$months = [

    1 => "มกราคม",

    2 => "กุมภาพันธ์",

    3 => "มีนาคม",

    4 => "เมษายน",

    5 => "พฤษภาคม",

    6 => "มิถุนายน",

    7 => "กรกฎาคม",

    8 => "สิงหาคม",

    9 => "กันยายน",

    10 => "ตุลาคม",

    11 => "พฤศจิกายน",

    12 => "ธันวาคม"

];

$province_list = [

    "สมุทรปราการ",

    "สมุทรสาคร",

    "สมุทรสงคราม",

    "เพชรบุรี",

    "ชลบุรี"

];

$selectedProvince = $_POST["province"] ?? "";

$selectedYear = $_POST["year"] ?? date("Y") + 543;

$selectedMonth = $_POST["month"] ?? date("n");

?>

<?php

if ($_SERVER["REQUEST_METHOD"] == "POST") {

    $province = trim($selectedProvince);

    $year = (int)$selectedYear;

    $month = (int)$selectedMonth;

    if ($province == "") {

        $error = "Please select province.";

    }
    elseif ($year <= 0) {

        $error = "Invalid year.";

    }
    elseif ($month < 1 || $month > 12) {

        $error = "Invalid month.";

    }
    else {

        $python = 'python';

        $script = realpath(
            "../model/Classification_RandomForest.py"
        );

        $province64 = base64_encode(
            $province
        );

        $command =
        
        escapeshellarg($python)
        . " " . 
        escapeshellarg($script)
        . " " . 
        escapeshellarg($province64)
        . " " .
        escapeshellarg($year)
        . " " . 
        escapeshellarg($month)
        . " --b64 2>&1";
        
        $output = shell_exec($command);

        if (!$output) {

    $error = "Python execution failed.";
    
    }
    
    else {

    $result = json_decode($output, true);

    if (json_last_error() !== JSON_ERROR_NONE) {

        $error = "Invalid JSON response: " . json_last_error_msg()
               . "<br><pre>" . htmlspecialchars($output) . "</pre>";

        $result = null;

    }
    elseif (
        isset($result["status"])
        &&
        $result["status"] == "error"
    ) {

        $error = $result["message"] ?? "Python returned an error.";

    }
    }
    }

}

?>

<!DOCTYPE html>
<html lang="th">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Classification Model</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet">

<link rel="stylesheet"
      href="../css/menu.css">

<link rel="stylesheet"
      href="../css/data.css">

</head>

<body>

<?php include "menu.php"; ?>

<div class="container page-container">

<div class="page-header mb-4">

<h2>

Random Forest Classification

</h2>

<p class="text-muted">

Predict Mackerel Density Level

</p>

</div>

<?php if($error != ""): ?>

<div class="alert alert-danger">

<?= htmlspecialchars($error) ?>

</div>

<?php endif; ?>

<div class="card shadow-sm border-0 mb-4">

<div class="card-header">

Prediction Parameters

</div>

<div class="card-body">

<form method="post">

<div class="row">

<div class="col-md-4 mb-3">

<label class="form-label">

Province

</label>

<select
    class="form-select"
    name="province"
    required
>

<option value="">

-- Select Province --

</option>

<?php foreach($province_list as $province): ?>

<option
    value="<?= htmlspecialchars($province) ?>"
    <?= ($selectedProvince == $province) ? "selected" : "" ?>
>

<?= htmlspecialchars($province) ?>

</option>

<?php endforeach; ?>

</select>

</div>

<div class="col-md-4 mb-3">

<label class="form-label">

Year

</label>

<input
    type="number"
    class="form-control"
    name="year"
    value="<?= htmlspecialchars($selectedYear) ?>"
    min="2560"
    max="2600"
    required
>

</div>

<div class="col-md-4 mb-3">

<label class="form-label">

Month

</label>

<select
    class="form-select"
    name="month"
    required
>

<?php foreach($months as $number => $name): ?>

<option
    value="<?= $number ?>"
    <?= ($selectedMonth == $number) ? "selected" : "" ?>
>

<?= htmlspecialchars($name) ?>

</option>

<?php endforeach; ?>

</select>

</div>

</div>

<div class="text-end">

<button
    type="submit"
    class="btn btn-success px-4"
>

Predict

</button>

</div>

</form>

</div>

</div>

<?php if($result && $result["status"] == "success"): ?>

<div class="card shadow-sm border-0 mb-4">

<div class="card-header bg-success text-white">

Classification Result

</div>

<div class="card-body text-center">

<span class="badge bg-<?= htmlspecialchars($result["badge_color"]) ?> fs-1 px-5 py-3">

<?= htmlspecialchars($result["level"]) ?>

</span>

<p class="mt-3 fs-5">

<?= htmlspecialchars($result["description"]) ?>

</p>

<hr>

<div class="row">

    <div class="col-md-3">

        <h6>
            Sea Surface Temperature
        </h6>

        <p class="fs-5">
            <?= number_format($result["sst"], 2) ?> °C
        </p>

    </div>

    <div class="col-md-3">

        <h6>
            Chlorophyll-a
        </h6>

        <p class="fs-5">
            <?= number_format($result["chlor_a"], 2) ?> mg/m³
        </p>

    </div>

    <div class="col-md-3">

        <h6>
            Rainfall
        </h6>

        <p class="fs-5">
            <?= number_format($result["rainfall"], 2) ?> mm
        </p>

    </div>

    <div class="col-md-3">

        <h6>
            Wind Speed
        </h6>

        <p class="fs-5">
            <?= number_format($result["wind_speed"], 2) ?> m/s
        </p>

    </div>

</div>

<div class="card shadow-sm border-0 mb-4">

    <div class="card-header">

        Model Performance

    </div>

    <div class="card-body">

        <div class="row text-center">

            <div class="col-md-3">

                <h6>Accuracy</h6>

                <p class="display-6 fw-bold text-success">

                    <?= number_format($result["accuracy"], 2) ?>%

                </p>

            </div>

            <div class="col-md-3">

                <h6>Precision</h6>

                <p class="display-6 fw-bold">

                    <?= number_format($result["precision"], 2) ?>%

                </p>

            </div>

            <div class="col-md-3">

                <h6>Recall</h6>

                <p class="display-6 fw-bold">

                    <?= number_format($result["recall"], 2) ?>%

                </p>

            </div>

            <div class="col-md-3">

                <h6>F1-Score</h6>

                <p class="display-6 fw-bold">

                    <?= number_format($result["f1"], 2) ?>%

                </p>

            </div>

        </div>

        <p class="text-muted text-center mb-0">

            Metrics from Test Dataset

        </p>

    </div>

</div>

<div class="card shadow-sm border-0 mb-4">

    <div class="card-header">

        Confusion Matrix

    </div>

    <div class="card-body">

        <?php
        $cm = $result["confusion_matrix"] ?? [];
        $labels = $result["confusion_labels"] ?? ["LOW", "MEDIUM", "HIGH"];
        ?>

        <?php if(count($cm) === 3): ?>

        <div class="table-responsive">

            <table class="table table-bordered text-center align-middle">

                <thead class="table-light">

                    <tr>

                        <th rowspan="2">

                            Actual \ Predicted

                        </th>

                        <th colspan="3">

                            Predicted

                        </th>

                    </tr>

                    <tr>

                        <?php foreach($labels as $label): ?>

                        <th>

                            <?= htmlspecialchars($label) ?>

                        </th>

                        <?php endforeach; ?>

                    </tr>

                </thead>

                <tbody>

                    <?php foreach($cm as $i => $row): ?>

                    <tr>

                        <th class="table-light">

                            <?= htmlspecialchars($labels[$i]) ?>

                        </th>

                        <?php foreach($row as $value): ?>

                        <td class="fs-5 fw-bold">

                            <?= (int)$value ?>

                        </td>

                        <?php endforeach; ?>

                    </tr>

                    <?php endforeach; ?>

                </tbody>

            </table>

        </div>

        <p class="text-muted text-center mb-0">

            Rows = Actual Class &nbsp; | &nbsp;
            Columns = Predicted Class

        </p>

        <?php else: ?>

        <div class="alert alert-warning mb-0">

            Confusion Matrix data not available.

        </div>

        <?php endif; ?>

    </div>

</div>

<div class="card shadow-sm border-0 mb-4">

<div class="card-header">

Prediction Probability

</div>

<div class="card-body">

<table class="table table-bordered text-center">

<thead>

<tr>

<th>

Density Level

</th>

<th>

Probability

</th>

</tr>

</thead>

<tbody>

<?php foreach($result["probability"] as $level => $value): ?>

<tr>

<td>

<?= htmlspecialchars($level) ?>

</td>

<td>

<?= number_format($value,2) ?> %

</td>

</tr>

<?php endforeach; ?>

</tbody>

</table>

</div>

</div>

<div class="card shadow-sm border-0 mb-4">

<div class="card-header">

Classification Probability Graph

</div>

<div class="card-body text-center">

<?php if(!empty($result["probability_graph"])): ?>

<img

    src="<?= htmlspecialchars($result["probability_graph"]) ?>?t=<?= time() ?>"

    class="img-fluid rounded shadow"

    alt="Probability Graph"

>

<?php else: ?>

<div class="alert alert-warning mb-0">

Probability graph not found.

</div>

<?php endif; ?>

</div>

</div>

<div class="card shadow-sm border-0 mb-4">

<div class="card-header">

Density Classification

</div>

<div class="card-body">

<table class="table table-bordered text-center">

<thead>

<tr>

<th>

Level

</th>

<th>

Description

</th>

</tr>

</thead>

<tbody>

<tr>

<td>

<span class="badge bg-success">

LOW

</span>

</td>

<td>

Low Mackerel Density Area

</td>

</tr>

<tr>

<td>

<span class="badge bg-warning text-dark">

MEDIUM

</span>

</td>

<td>

Medium Mackerel Density Area

</td>

</tr>

<tr>

<td>

<span class="badge bg-danger">

HIGH

</span>

</td>

<td>

High Mackerel Density Area

</td>

</tr>

</tbody>

</table>

</div>

</div>

<?php endif; ?>

</div>

<?php include "footer.php"; ?>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

</body>

</html>