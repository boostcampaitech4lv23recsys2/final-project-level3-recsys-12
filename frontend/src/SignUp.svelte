<script>
	import { link, push } from 'svelte-spa-router'
    import ImageBlock from './SignUpElement/ImageBlock.svelte';
    import {setContext, getContext} from 'svelte'
    
	// import { email } from './SignUpElement/SignUpForm.svelte'
    let house_list = []

    let email
    let is_success

    async function isEmailDup(e){
        const next_btn = document.querySelector("#next_button")
        const next_btn_disable = document.querySelector("#next_button_disable")
        e.preventDefault()
        if (email == null){
            alert("이메일을 입력해주세요.")
        }else{
            let url = "http://127.0.0.1:8000/signup/"+email
            await fetch(url).then((response) => {
                response.json().then((json) => {
                    if (response.status == 400){
                        is_success = false
                        alert("중복된 이메일입니다.")
                        next_btn.classList.add("deactivate")
                        next_btn.classList.remove("activate")
                        next_btn_disable.classList.add("activate")
                        next_btn_disable.classList.remove("deactivate")
                    }else{
                        is_success = true
                        alert("사용할 수 있는 이메일입니다.")
                        next_btn.classList.add("activate")
                        next_btn.classList.remove("deactivate")
                        next_btn_disable.classList.add("deactivate")
                        next_btn_disable.classList.remove("activate")
                    }
                })
            })
            console.log(selected_cnt)
        }
    }

    function goToScroll(e) {
        e.preventDefault()
        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location, behavior: 'smooth'});
    }

	async function get_items() {
		await fetch("http://127.0.0.1:8000/signup").then((response) => {
			response.json().then((json) => {
				house_list = json
			})
		})
		.catch((error) => console.log(error))
	}

    async function post_member(){
        let url = "http://127.0.0.1:8000/member/"+email+"/"+JSON.stringify(Array.from(selected_img))
        let params = {
            "member_email" : email,
            "selected_house_id" : JSON.stringify(Array.from(selected_img))
        }
        // let options = {
        //     method: "post",
        //     headers: {
        //         "Content-Type": 'application/json'
        //     },
        //     body: JSON.stringify(params)
        // }
        // console.log(options.body)
        // console.log(typeof(options.body))
        
        await fetch(url).then((response)=>{
            if(response.status >= 200 && response.status < 300){
                push("/login-user")
                console.log("회원가입 성공!")
            }else{
                alert("회원가입에 실패하였습니다.")
                push("/signup-user")
            }
        })
        // await fetch(url,options).then((response)=>{
        //     if(response.status >= 200 && response.status < 300){
        //         push("/login-user")
        //     }else{
        //         alert("회원가입에 실패하였습니다.")
        //         push("/signup-user")
        //     }
        // })
    }

    async function post_inference_result() {
        let url = "http://127.0.0.1:8000/insert-inference-result"
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
            <form>
                <input type="email" id="email_form" placeholder="E-mail 주소를 입력하세요." bind:value="{email}">
                <button id="dupcheck" on:click={isEmailDup}>중복 확인</button>
                <button class="h" on:click={goToScroll}>Next!</button>
            </form>
        </div>
        <div class="container_title">
            <br>
            <h4 style="text-align: center; margin:0;">마음에 드는 이미지를 선택해 주세요.</h4>
            <br>
            <p style="text-align: center;">많이 선택할수록 개선된 결과가 표시됩니다.</p>
            <br>
        </div>
		<!-- <div class="container_wrapper row gx-3 gx-lg-3 row-cols-3 row-cols-md-3 row-cols-xl-4 justify-content-center"> -->
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
        <button id="next_button" class="prevent_btn nextbtn deactivate" disabled on:click={next_btn_click}>
            <div id="selectbtn_wrapper">
                <span>최소 5개 선택해 주세요.</span>
                <span></span>
                <span id="selected_num">Next!({selected_cnt}/5)</span>
            </div>
        </button>
        <div id="next_button_disable" class="prevent_btn nextbtn activate">중복확인 먼저 진행해 주세요.</div>
	</div>
</section>


<style>
.container_wrapper{
	/* width: 100%; */
	padding: 10px 10px;
	display: flex;
	flex-flow: row wrap;
	justify-content: center;
	align-items: center;
	gap: 10px;
	/* border: 1px solid gray;
	border-radius: 30px; */
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
#dupcheck{
    scroll-behavior: smooth;
}
.deactivate{
    visibility: hidden;
    z-index: -1;
}
.activate{
    z-index: 1;
}
#next_button_disable{
    display: flex;
    color: white;
    justify-content: center;
    align-items: center;
}
</style>