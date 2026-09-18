<?php

/*
 * Session ต้องเริ่มก่อนอ่าน $_SESSION
 */
if (session_status() === PHP_SESSION_NONE)
{
    session_start();
}

require_once('../config/class.connect.php');

?>


<nav class="navbar navbar-expand-lg bg-body-tertiary">

    <div class="container-fluid">

        <a
            class="navbar-brand"
            href="../frontend/home.php"
        >
            Project
        </a>


        <button
            class="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
        >

            <span class="navbar-toggler-icon"></span>

        </button>


        <div
            class="collapse navbar-collapse"
            id="navbarSupportedContent"
        >

            <ul class="navbar-nav me-auto mb-2 mb-lg-0">


<?php

/*
 * ตรวจสอบว่า Login แล้วหรือยัง
 */
if (
    isset($_SESSION['uid']) &&
    $_SESSION['uid'] != ''
)
{

    $uid = (int)$_SESSION['uid'];


    /*
     * ดึง Menu ตาม User
     */
    $sql = "
        SELECT
            app.id,
            app.name,
            app.dir,
            app.detail,
            app.appgroup,
            MAX(acl.accl) AS accl

        FROM app

        INNER JOIN acl
            ON acl.appid = app.id

        INNER JOIN uig
            ON uig.ugid = acl.ugid

        WHERE
            app.status = 1
            AND acl.status = 1
            AND uig.status = 1
            AND uig.uid = ?
            AND acl.accl > 0

        GROUP BY
            app.id,
            app.name,
            app.dir,
            app.detail,
            app.appgroup

        ORDER BY
            app.appgroup ASC,
            app.id ASC
    ";


    $conn = new connect();

    $res = $conn->query(
        $sql,
        array($uid)
    );


    while ($cdr = $res->fetch())
    {

        echo '<li class="nav-item">';

        echo '<a
                class="nav-link"
                href="index.php?option='
                . htmlspecialchars($cdr['dir'], ENT_QUOTES, 'UTF-8')
                . '&task=def"
              >';

        echo htmlspecialchars(
            $cdr['name'],
            ENT_QUOTES,
            'UTF-8'
        );

        echo '</a>';

        echo '</li>';
    }

?>


                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="../frontend/home.php"
                    >
                        Homepage
                    </a>

                </li>


                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="index.php?option=logs&task=logout"
                    >
                        Log out
                    </a>

                </li>


<?php

}
else
{

?>


                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="index.php?option=logs&task=login_form"
                    >
                        Log in
                    </a>

                </li>


<?php

}

?>


            </ul>

        </div>

    </div>

</nav>