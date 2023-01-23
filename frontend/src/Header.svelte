<script>

    import { link } from 'svelte-spa-router'
    import { access_token, member_email, is_login } from './store'
    const logo = document.querySelector("#logo")
    function onLogoHover(e){
        if (e.target.textContent == "@_@") {
            e.target.textContent = "@..@"
        } else {
            e.target.textContent = "@_@"
        }
    }

    function logout() {
        $access_token = ''
        $member_email = ''
        $is_login = false
    }

</script>

<header class="header">
    <nav class="navbar navbar-expand-lg navbar-light navbar-bg">
        <div class="container px-4 px-lg-5">
            <a class="navbar-brand" id="logo" on:mouseover={onLogoHover} on:mouseleave={onLogoHover} href="/">@_@</a>
            {#if ($is_login)}
            <a use:link href="/login" on:click={logout}>
                <button class="login-button" type="button">로그아웃</button>
            </a>
            {:else}
            <a use:link href="/login">
                <button class="login-button" type="button">로그인</button>
            </a>
            {/if}
        </div>
    </nav>
</header>

<style>
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
</style>