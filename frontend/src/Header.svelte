<script>

    import { link } from 'svelte-spa-router'
    import { access_token, member_email, is_login } from './store'

    function logout() {
        $access_token = ''
        $member_email = ''
        $is_login = false
    }

</script>

<header class="sticky-top header">
    <nav class="navbar navbar-expand-lg navbar-light navbar-bg">
        <div class="container px-4 px-lg-5">
            <a class="navbar-brand" id="logo" href="/">
                <img class="logo-image" src="https://user-images.githubusercontent.com/80964004/216566569-be28e4c4-91df-406e-b51e-db0140ce54fe.png" alt="...">
            </a>

            <span>
                {#if ($is_login)}
                <a use:link href="/mypage">
                    <button class="mypage-button" type="button">마이페이지</button>
                </a>
                <a use:link href="/login-user" on:click={logout}>
                    <button class="login-button" type="button">로그아웃</button>
                </a>
                {:else}
                <a use:link href="/login-user">
                    <button class="login-button" type="button">로그인</button>
                </a>
                {/if}
            </span>
        </div>
    </nav>
</header>

<style>

    .logo-image {
        height: 80px;
        margin-bottom: 10px;
    }
    header {
        height: 120px;
    }
    .navbar {
        border-bottom: solid 1px rgb(158, 158, 158);
    }
    /* navbar 배경 색상 */
    .navbar-bg {
        --bs-bg-opacity: 1;
        background-color: #FFFFFF;
    }

    /* 페이지 로고(RAYN) 크기 조절 */
    .navbar-brand {
        font-size: 55px;
    }
    .navbar-brand:hover {
        font-size: 55px;
        text-decoration: none;
    }

    /* 로그인 버튼 */
	.login-button {
		padding: 0.25rem 0.75rem;
		font-size: 1.25rem;
		line-height: 1;
		background-color: transparent;
		border: 1px solid transparent;
		border-radius: 0.25rem;
		transition: box-shadow 0.15s ease-in-out;
	}

    /* 로그인 버튼에 마우스 오버 시 테두리 */
    .login-button {
        transition: all .5s;
    }
    .login-button:hover {
        box-shadow: 0 0 0 3px black;
    }

    /* 마이페이지 버튼 */
	.mypage-button {
		padding: 0.25rem 0.75rem;
		font-size: 1.25rem;
		line-height: 1;
		background-color: transparent;
		border: 1px solid transparent;
		border-radius: 0.25rem;
		transition: box-shadow 0.15s ease-in-out;
	}

    /* 마이페이지 버튼에 마우스 오버 시 테두리 */
    .mypage-button {
        transition: all .5s;
    }
    .mypage-button:hover {
        box-shadow: 0 0 0 3px black;
    }
</style>