<script>
	import { push } from 'svelte-spa-router'
    import ImageBlock from './SignUpElement/ImageBlock.svelte';
    import { setContext } from 'svelte'
    
    let house_list = []
    let email, is_success, use_email = false
    async function isEmailDup(e){
        e.preventDefault()
        if (email == null) {
            is_success = 2  // 비어있는 이메일 입력
        }else {
            let url = import.meta.env.VITE_SERVER_URL + "/signup"
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

            await fetch(url,options).then((response) => {
                response.json().then((json) => {
                    if (response.status == 400){
                        is_success = 0  // 중복된 이메일
                    }else {
                        // let reg_exp = '[0-9a-zA-Z]{1,}@[0-9a-zA-Z]{1,}.[.]([0-9a-zA-Z]{2,}[-_.]?){1,}'
                        let reg_exp = /^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*.[a-zA-Z]{2,3}$/i
                        let regex = new RegExp(reg_exp);
                        if (regex.test(email)) {
                            is_success = 3
                        }else {
                            is_success = 1  // 형식이 맞지 않는 이메일
                        }
                    }
                })
            })
        }
    }
    function change_use_email(e) {
        e.preventDefault()
        if (use_email) {
            use_email = false;
        }else {
            use_email = true;
        }
    }

    function goToScroll(e) {
        e.preventDefault()
        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location, behavior: 'smooth'});
    }
    function goToTop(e) {
        e.preventDefault()
        window.scrollTo(0, 0)
    }

	async function get_items() {
        let url = import.meta.env.VITE_SERVER_URL + "/card"
		await fetch(url).then((response) => {
			response.json().then((json) => {
				house_list = json
			})
		})
		.catch((error) => console.log(error))
	}
    
    async function post_member(){
        let url = import.meta.env.VITE_SERVER_URL + "/member"
        let params = {
            "member_email" : email,
            "selected_house_id" : JSON.stringify(Array.from(selected_img))
        }
        let options = {
            method: "post",
            headers: {
                "Content-Type": 'application/json'
            },
            body: JSON.stringify(params)
        }

        await fetch(url,options).then((response)=>{
            if(response.status >= 200 && response.status < 300){
                push("/login-user")
                alert("회원가입에 성공하였습니다.")
            }else{
                alert("회원가입에 실패하였습니다.")
                push("/signup-user")
            }
        })
    }

    async function post_inference_result() {
        let url = import.meta.env.VITE_SERVER_URL + "/insert-inference-result"
        let params = {
            "member_email" : email
        }
        let options = {
            method: "post",
            headers: {
                "Content-Type": 'application/json'
            },
            body: JSON.stringify(params)
        }
        await fetch(url, options).then((response) => {
            response.json().then((json) => {
                if(json == "success") {
                    console.log("저장 완료!")
                } else {
                    console.log("저장 실패")
                }
            })
        })
    }

    function next_btn_click() {
        post_member()
        post_inference_result()
    }


	get_items()

	let selected_img = new Set();
	setContext("selected_img",selected_img);

    let selected_cnt = 0;
    setContext("selected_cnt",selected_cnt);

    setContext("is_success",is_success)
</script>

<hr>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">

<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
        
        <div id="signup_form_wrapper">
            <form class="input-form">
                {#if is_success == 3 && !use_email}
                <input type="email" id="email_form" class="form-control form-control-lg" placeholder="E-mail 주소를 입력하세요." bind:value="{email}" disabled>
                {:else}
                <input type="email" id="email_form" class="form-control form-control-lg" placeholder="E-mail 주소를 입력하세요." bind:value="{email}">
                {/if}
                {#if is_success == 3}                
                <button id="dupcheck" on:click={change_use_email} class="signup-button btn btn-info btn-block">이메일 변경</button>
                {:else}
                <button id="dupcheck" on:click={isEmailDup} class="signup-button btn btn-info btn-block">이메일 확인</button>
                {/if}
                {#if is_success == 3 && !use_email}
                <button class="next-button signup-button btn btn-info" on:click={goToScroll}>Next!</button>
                {:else}
                <button class="next-button signup-button btn btn-info" disabled on:click={goToScroll}>Next!</button>
                {/if}
            </form>
            <div>
                {#if is_success == 0}   <!-- 0 : 중복된 이메일 -->
                <div class="email-wrong">중복된 이메일입니다! 다른 이메일을 입력해주세요.</div>
                {:else if is_success == 1}  <!-- 1 : 잘못된 형식의 이메일 -->
                <div class="email-wrong">잘못된 형식의 이메일입니다! <br>(예: example@naver.com)</div>
                {:else if is_success == 2}  <!-- 2 : 비어있는 이메일 -->
                <div class="email-wrong">이메일을 입력해주세요.</div>
                {:else if is_success == 3} <!-- 3 : 올바른 이메일 -->
                <div id="email-check-done" class="email-right">이메일이 확인되었습니다. <br>[Next!] 버튼을 눌러주세요!</div>
                {:else}
                <div class="email-wrong">이메일 중복 확인을 진행해주세요!</div>
                {/if}
            </div>
        </div>
        {#if is_success == 3 && !use_email}
        <div class="container_title">
            <br>
            <h4 style="text-align: center; margin:0;">마음에 드는 이미지를 선택해 주세요.</h4>
            <br>
            <p style="text-align: center;">많이 선택할수록 개선된 결과가 표시됩니다.</p>
            <br>
        </div>
        <div class="container_wrapper justify-content-center">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->

			<!-- item_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
            {#each house_list as item}
            <ImageBlock {item}/>
            {/each}
		</div>
        <button id="next_button" class="prevent_btn nextbtn" on:click={next_btn_click}>
            <div id="selectbtn_wrapper">
                <span>최소 5개 선택해 주세요.</span>
                <span></span>
                <span id="selected_num">Next!({selected_cnt}/5)</span>
            </div>
        </button>
        {/if}
    </div>
    <div class="go-top-button">
        <button class="btn btn-secondary" on:click="{goToTop}">
            <img class="top-button" src="https://cdn-icons-png.flaticon.com/512/130/130906.png" alt="...">
            <br>Top
        </button>
    </div>
</section>


<style>
    .go-top-button {
        position: fixed;
        right: 5%;
        bottom: 100px;
    }
    .top-button {
        width: 20px;
        
    }
    .input-form {
        display: flex;
    }
    .signup-button {
        color: white;
        font-size: 20px;
        width: 100px;
        height: 50px;
        margin-top: 10px;
    }
    .container_wrapper{
        padding: 10px 10px;
        display: flex;
        flex-flow: row wrap;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }
    .prevent_btn{
        opacity: 0.8;
    }
    .nextbtn{
        position: fixed;
        width: 100%;
        height: 10vh;
        border: 0;
        color: white;
        bottom: 0px;
        left: 0;
        background-color: #333;
    }
    .nextbtn:hover{
        background-color: gray;
        transition: background-color 0.5s;
    }
    #selectbtn_wrapper{
        display:flex;
        justify-content: space-evenly;
        opacity: 1;
    }
    .container_title{
        background-color: black;
        color: white;
        margin: 0px;
        padding: 0px;
    }
    p{
        margin: 0px;
        padding: 0px;
    }
    #signup_form_wrapper{
        display: flex;
        flex-flow: column nowrap;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    #email_form {
        margin: 10px;
        font-size: 100%;
    }
    #dupcheck{
        scroll-behavior: smooth;
        width: 50%;
        margin: 10px;
        font-size: 100%;
    }
    .next-button {
        font-size: 100%;
    }
    .email-wrong {
        color: red;
        font-weight: bold;
    }
    .email-right {
        color: green;
        font-weight: bold;
        text-align: center;
    }
</style>