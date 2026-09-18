<?php

session_start();

echo "<h2>Railway Session Test</h2>";

echo "<pre>";
print_r($_SESSION);
echo "</pre>";

echo "<hr>";

echo "Session ID: " . session_id();

echo "<br><br>";

if (isset($_SESSION['uid']))
{
    echo "<strong>UID = " . $_SESSION['uid'] . "</strong>";
}
else
{
    echo "<strong>UID = NOT SET</strong>";
}