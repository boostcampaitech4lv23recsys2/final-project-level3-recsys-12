<script>
	import { link, push } from 'svelte-spa-router'
    import { access_token, member_email, is_login } from './store'

    let email = ""
    async function login(event) {
        event.preventDefault()
        let url = import.meta.env.VITE_SERVER_URL + "/login"
        let params = {
            "member_email" : email,
        }
        let options = {
            method: "post",
            headers: {
                "Content-Type": 'application/json'
            },
            body: JSON.stringify(params)
        }
        if (check_email()) {
            await fetch(url,options).then((response) => {
                response.json().then(json => {
                    if (response.status >= 200 && response.status < 300) {  // 200 ~ 299
                        $access_token = json.access_token
                        $member_email = json.member_email
                        $is_login = true
                        push('/')
                    } else if (response.status === 401) { // token time out
                        access_token.set('')
                        member_email.set('')
                        is_login.set('')
                        alert("로그인이 필요합니다.")
                        push('/login-user')
                    }
                    else {
                        alert("존재하지 않는 이메일입니다.")
                        push('/login-user')
                    }
                })
                .catch(error => {
                    console.log("error")
                    alert(JSON.stringify(error))
                })
            })
        }else {
            alert("올바른 이메일 형식을 입력해주세요.")
        }
    }
        
    function check_email() {
        // let reg_exp = '[0-9a-zA-Z]{1,}@[0-9a-zA-Z]{1,}.[.]([0-9a-zA-Z]{2,}[-_.]?){1,}'
        let reg_exp = /^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*.[a-zA-Z]{2,3}$/i
        let regex = new RegExp(reg_exp);
        if (regex.test(email)) {
            return true;
        }
        return false;
    }

    function enter_login(e) {
        e.preventDefault()
        if (window.event.keyCode == 13) {
            login(e)
        }
    }
</script>

<section>
    <div class="container-fluid">
    
        <form style="width: 23rem;" on:submit={login}>

            <h3 class="fw-normal mb-3 pb-3" style="letter-spacing: 1px;">Login</h3>

            <div class="form-outline mb-4">
                <label class="form-label" for="form2Example18">회원가입 시 사용한 이메일을 입력해주세요</label>
                <input on:keyup="{enter_login}" type="email" id="form2Example18" bind:value={email} class="form-control form-control-lg" placeholder="이메일"/>
                <button on:click="{login}" class="login-button btn btn-info btn-lg btn-block" type="button">Login</button>
            </div>

            <p class="bottom-link small mb-5 pb-lg-2">
                <a use:link class="each-link" href="/signup-user">회원가입</a>
            </p>
        </form>
    </div>
</section>

<style>

    section {
        padding-top: 50px;
        height:calc(100vh - 240px); /* 전체 height - (header + footer)px*/
    }
    .container-fluid {
        display: flex;
        justify-content: center;
    }

    .login-button {
        margin-top: 2%;
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