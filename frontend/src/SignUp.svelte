<script>
	import { push } from 'svelte-spa-router'
    import ImageBlock from './SignUpElement/ImageBlock.svelte';
    import { setContext } from 'svelte'
    import TopButton from './GoTop.svelte'
    
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
	    get_items()
    }
    function change_use_email(e) {
        e.preventDefault()
        if (use_email) {
            use_email = false;
            is_success = 0;
            isEmailDup(e);
        }else {
            use_email = true;
        }
    }

    function goToScroll(e) {
        e.preventDefault()
        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location-120, behavior: 'smooth'});
        
    }

    let card_id_list = []
    
    let space_value, family_value, house_size = 0
	async function get_items() {
        let url = import.meta.env.VITE_SERVER_URL + "/card"
        let params = {
                "card_id_list": JSON.stringify(Array.from(selected_img)),
                "space": JSON.stringify(space_value),
                "size": JSON.stringify(house_size),
                "family": JSON.stringify(family_value)
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
				card_id_list = json
			})
		})
		.catch((error) => console.log(error))
	}

    async function post_member(){
        let url = import.meta.env.VITE_SERVER_URL + "/member"
        let params = {
            "member_email" : email,
            "selected_card_id" : JSON.stringify(Array.from(selected_img))
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
        let response = confirm("회원가입을 완료하시겠습니까?")
        if (response) {
            post_member()
            post_inference_result()
        }
    }

	let selected_img = new Set();
	setContext("selected_img",selected_img);

    let selected_cnt = 0;
    setContext("selected_cnt",selected_cnt);

    setContext("is_success",is_success)

    let next_cnt = 1
    function get_next_items() {
        get_items()
        next_cnt += 1

        var location = document.querySelector(".container_title").offsetTop;
        window.scrollTo({top: location-120, behavior: 'smooth'});
        const house_img_list = document.getElementsByClassName("house")
        for (let house_img of house_img_list) {
            house_img.style.opacity = 1
            house_img.style.borderWidth = "0px"
        }
    }
    
</script>

<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">

<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
        <div id="signup_form_wrapper">
            <div class="description">추천받고 싶은 가구를 배치할 집의 형태를 입력해주세요!</div>
            <form>
                <div class="input-area-wrapper">
                    <label for="place" class="col-sm-2 col-form-label input-area-label">공간</label>
                    <select id="place" class="form-select input-area" required bind:value={space_value}>
                        <option value=null selected>공간을 선택하세요.</option>
                        <option value="원룸&오피스텔">원룸&오피스텔</option>
                        <option value="아파트">아파트</option>
                        <option value="빌라&연립">빌라&연립</option>
                        <option value="단독주택">단독주택</option>
                        <option value="사무공간">사무공간</option>
                        <option value="상업공간">상업공간</option>
                        <option value="기타">기타</option>
                    </select>
                </div>

                <div class="input-area-wrapper">
                    <label for="family" class="col-sm-2 col-form-label input-area-label">가족형태</label>
                    <select id="family" class="form-select input-area" required bind:value={family_value}>
                        <option value=null selected>가족형태를 선택하세요.</option>
                        <option value="싱글라이프">싱글라이프</option>
                        <option value="신혼 부부">신혼 부부</option>
                        <option value="아기가 있는 집">아기가 있는 집</option>
                        <option value="취학 자녀가 있는 집">취학 자녀가 있는 집</option>
                        <option value="부모님과 함께 사는 집">부모님과 함께 사는 집</option>
                        <option value="기타">기타</option>
                    </select>
                </div>

                <div class="input-area-wrapper">
                    <label for="size" class="col-sm-2 col-form-label input-area-label">평수</label>
                    <input type="number" id="size" class="form-control input-area" placeholder="평수를 입력하세요." required bind:value={house_size}>
                </div>

                <div class="description-login">로그인 시 사용할 이메일을 입력해주세요.</div>
                <div class="input-form">
                    
                    {#if is_success == 3 && !use_email}
                    <input type="email" id="email_form" class="form-control" placeholder="E-mail 주소를 입력하세요." bind:value="{email}" disabled>
                    {:else}
                    <input type="email" id="email_form" class="form-control form-control-lg" placeholder="E-mail 주소를 입력하세요." bind:value="{email}">
                    {/if}
                    {#if is_success == 3}                
                    <button id="dupcheck" on:click={change_use_email} class="signup-button btn btn-info btn-block">이메일 변경</button>
                    {:else}
                    <button id="dupcheck" on:click={isEmailDup} class="signup-button btn btn-info btn-block">이메일 확인</button>
                    {/if}
                    {#if is_success == 3 && !use_email && space_value!=null && family_value!=null && house_size!=-1}
                    <button class="next-button signup-button btn btn-info" on:click={goToScroll}>Next!</button>
                    {:else}
                    <button class="next-button signup-button btn btn-info" disabled on:click={goToScroll}>Next!</button>
                    {/if}
                </div>
            </form>
            <div>
                {#if is_success == 0}   <!-- 0 : 중복된 이메일 -->
                <div class="email-wrong">중복된 이메일입니다! 다른 이메일을 입력해주세요.</div>
                {:else if is_success == 1}  <!-- 1 : 잘못된 형식의 이메일 -->
                <div class="email-wrong">잘못된 형식의 이메일입니다! <br>(예: example@naver.com)</div>
                {:else if is_success == 2}  <!-- 2 : 비어있는 이메일 -->
                <div class="email-wrong">이메일을 입력해주세요.</div>
                {:else if is_success == 3} <!-- 3 : 올바른 이메일 -->
                {#if space_value==null || family_value==null || house_size==0}
                    <div class="email-wrong">모든 정보를 입력해주세요.</div>
                    {:else}
                    <div id="email-check-done" class="email-right">[Next!] 버튼을 눌러주세요!</div>
                    {/if}
                {:else}
                <div class="email-wrong">이메일 중복 확인을 진행해주세요!</div>
                {/if}
            </div>
        </div>
        {#if is_success == 3 && !use_email && space_value!=null && family_value!=null && house_size!=-1}
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
            {#each card_id_list as item}
            <ImageBlock {item}/>
            {/each}
            <br>
            
		</div>
        {#if next_cnt < 5}
            <button class="btn btn-secondary btn-block nextbtn-top" on:click={get_next_items} style="width:100%">더 많은 상품 보기 ></button>
            {:else}
            <button class="btn btn-secondary btn-block nextbnt-top-finish" on:click={get_next_items} style="width:100%;">다음 상품들 보기 ><br>상품 선택을 완료하셨다면 아래를 클릭해 회원가입을 완료해주세요!</button>
            <!-- <button class="btn btn-secondary btn-block nextbnt-top-finish" on:click={get_next_items} style="width:100%;background-color:white;color:black" disabled>하단의 [Next!]버튼을 클릭해 회원가입을 완료해주세요!</button> -->
            {/if}
        <button id="next_button" class="prevent_btn nextbtn" on:click={next_btn_click}>
            <div id="selectbtn_wrapper">
                <span>최소 5개 선택해 주세요.</span>
                <span></span>
                <span id="selected_num">회원가입 완료하기!({selected_cnt}/5)</span>
            </div>
        </button>
        
        <div class="go-top-button">
            <TopButton />
        </div>
        {/if}
    </div>
</section>


<style>
    .description {
        font-size: 1.3rem;
        font-weight: bold;
        margin: 30px;
    }
    .description-login {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 50px;
        text-align: center;
    }
    .input-area-wrapper {
        display: flex;
        margin-bottom: 25px;
    }
    .input-area-label {
        width: 20%;
        font-weight: bold;
        text-align: center;
    }
    .input-area {
        width: 80%;
    }
    .go-top-button {
        position: fixed;
        right: 5%;
        bottom: 100px;
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
    .nextbtn-top {
        position: fixed;
        width: 100%;
        height: 5vh;
        border: 0;
        color: white;
        bottom: 8.01vh;
        left: 0;
    }
    .nextbnt-top-finish {
        position: fixed;
        width: 100%;
        height: 8vh;
        border: 0;
        color: white;
        bottom: 8.01vh;
        left: 0;
    }
    .nextbtn{
        position: fixed;
        width: 100%;
        height: 8vh;
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
        height: calc(100vh - 120px);
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