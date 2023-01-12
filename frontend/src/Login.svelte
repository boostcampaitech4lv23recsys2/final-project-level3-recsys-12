<script>
	import { link } from 'svelte-spa-router'
</script>

<hr>

<section class="vh-100">
    <div class="container-fluid">
    
        <form style="width: 23rem;">

            <h3 class="fw-normal mb-3 pb-3" style="letter-spacing: 1px;">Log in</h3>

            <div class="form-outline mb-4">
                <label class="form-label" for="form2Example18">회원가입 시 사용한 닉네임을 입력해주세요</label>
                <input type="email" id="form2Example18" class="form-control form-control-lg" placeholder="닉네임"/>
            </div>

            <div class=" pt-1 mb-4">
                <button class="login-button btn btn-info btn-lg btn-block" type="button">Login</button>
            </div>

            <p class="bottom-link small mb-5 pb-lg-2">
                <a class="each-link" href="#!">닉네임 재설정</a>&nbsp;&nbsp;&nbsp;
                <a use:link class="each-link" href="/signup">회원가입</a>
            </p>
        </form>
    </div>
</section>

<style>

    .vh-100 {
        padding-top: 50px;
    }

    .container-fluid {
        display: flex;
        justify-content: center;
    }

    .login-button {
        width: 100%;
        color: white;
    }

    .bottom-link {
        text-align: center;
    }

    .each-link {
        text-decoration: none;
        font-size: 1.1rem;
        transition: transform .2s;
        font-weight: normal;
        color: black;
    }

    .each-link:hover {
        font-weight: bold;
        color: black;
    }

</style>