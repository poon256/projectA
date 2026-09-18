<?php

ob_start();

/*
 * Session Configuration
 */
session_set_cookie_params([
    'lifetime' => 0,
    'path' => '/',
    'secure' => (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off'),
    'httponly' => true,
    'samesite' => 'Lax'
]);

session_start();


/*
 * Option
 */
if (isset($_REQUEST['option']) && $_REQUEST['option'] != '')
{
    $option = $_REQUEST['option'];
}
elseif (isset($_SESSION['uid']))
{
    $option = "users";
}
else
{
    $option = "logs";
}


/*
 * Task
 */
if (isset($_REQUEST['task']) && $_REQUEST['task'] != '')
{
    $task = $_REQUEST['task'];
}
elseif (isset($_SESSION['uid']))
{
    $task = "def";
}
else
{
    $task = "login_form";
}

?>
<!DOCTYPE html>

<html>

<head>

    <title>
        Mackerel Engine - Backend
    </title>

    <meta
        http-equiv="Content-Type"
        content="text/html; charset=utf-8"
    />

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    />

    <link
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet"
    >

    <link
        rel="stylesheet"
        href="https://code.jquery.com/ui/1.14.0/themes/base/jquery-ui.css"
    >

    <link
        rel="stylesheet"
        href="https://cdn.datatables.net/2.1.6/css/dataTables.bootstrap5.min.css"
    >

</head>


<body>

<?php

require_once('../config/class.connect.php');


/*
 * ถ้ายังไม่ได้ Login
 * และพยายามเข้า Module อื่น
 * ให้กลับไป Login
 */
if (
    !isset($_SESSION['uid']) &&
    $option != 'logs'
)
{
    $option = 'logs';
    $task = 'login_form';
}


/*
 * ตรวจสอบว่า Class มีจริง
 */
$classFile = 'class.' . $option . '.php';

if (file_exists($classFile))
{
    require_once($classFile);

    if (class_exists($option))
    {
        $clas = new $option();

        if (method_exists($clas, $task))
        {
            $clas->$task();
        }
        else
        {
            echo "<div class='container mt-5'>";
            echo "<div class='alert alert-danger'>";
            echo "ไม่พบ Task: " . htmlspecialchars($task);
            echo "</div>";
            echo "</div>";
        }
    }
    else
    {
        echo "<div class='container mt-5'>";
        echo "<div class='alert alert-danger'>";
        echo "ไม่พบ Class: " . htmlspecialchars($option);
        echo "</div>";
        echo "</div>";
    }
}
else
{
    echo "<div class='container mt-5'>";
    echo "<div class='alert alert-danger'>";
    echo "ไม่พบ Module: " . htmlspecialchars($option);
    echo "</div>";
    echo "</div>";
}

?>

</body>


<script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
></script>

<script
    src="https://code.jquery.com/jquery-3.7.1.js"
></script>

<script
    src="https://code.jquery.com/ui/1.14.0/jquery-ui.js"
></script>

<script
    src="https://cdn.datatables.net/2.1.6/js/dataTables.min.js"
></script>

<script
    src="https://cdn.datatables.net/2.1.6/js/dataTables.bootstrap5.min.js"
></script>

<script src="project.js"></script>

</html>