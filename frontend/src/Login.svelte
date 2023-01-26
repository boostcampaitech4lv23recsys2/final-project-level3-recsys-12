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
        console.log(options.body)
        console.log(typeof(options.body))

        await fetch(url,options).then((response)=>{
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
            }
        )}
        
</script>

<hr>

<section class="vh-100">
    <div class="container-fluid">
    
        <form style="width: 23rem;">

            <h3 class="fw-normal mb-3 pb-3" style="letter-spacing: 1px;">Login</h3>

            <div class="form-outline mb-4">
                <label class="form-label" for="form2Example18">회원가입 시 사용한 이메일을 입력해주세요</label>
                <input type="email" id="form2Example18" bind:value={email} class="form-control form-control-lg" placeholder="이메일"/>
            </div>

            <div class=" pt-1 mb-4">
                <button on:click="{login}" class="login-button btn btn-info btn-lg btn-block" type="button">Login</button>
            </div>

            <p class="bottom-link small mb-5 pb-lg-2">
                <a use:link class="each-link" href="/signup-user">회원가입</a>
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