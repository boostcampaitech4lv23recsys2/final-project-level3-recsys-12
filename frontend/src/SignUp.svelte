<script>
	import { link, push } from 'svelte-spa-router'
    import ImageBlock from './SignUpElement/ImageBlock.svelte';
    import SignUpForm from './SignUpElement/SignUpForm.svelte';
    import {getContext, setContext} from "svelte"
	
    let house_list = []
    let email = getContext("email")

	async function get_items() {
		await fetch("http://localhost:8000/signup").then((response) => {
			response.json().then((json) => {
				house_list = json
			})
		})
		.catch((error) => console.log(error))
        // console.log(house_list)
	}

    async function post_member(e){
        e.preventDefault()
        let url = "http://localhost:8000/member/success"
        let params = {
            "member_email" : email,
            "selected_house_id" : Array.from(selected_img)
        }
        let options = {
            method: "post",
            headers: {
                "Content-Type": 'application/json'
            },
            body: JSON.stringify(params)
        }
        console.log(options.body)
        // console.log(typeof(options.body))
        
        await fetch(url,options).then((response)=>{
            if(response.status >= 200 && response.status < 300){
                push("/login")
            }else{
                alert("회원가입에 실패하였습니다.")
                push("/signup")
            }
        })
    }

	get_items()

	let selected_img = new Set();
	setContext("selected_img",selected_img);

    let selected_cnt = 0;
    setContext("selected_cnt",selected_cnt);
    let is_success = getContext("is_success")
    console.log(is_success)
</script>


<hr>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">


<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
        <SignUpForm/>
        <div class="container_title">
            <br>
            <h4 style="text-align: center; margin:0;">마음에 드는 이미지를 선택해 주세요.</h4>
            <br>
            <p style="text-align: center;">많이 선택할수록 개선된 결과가 표시됩니다.</p>
            <br>
        </div>
		<div class="container_wrapper row gx-3 gx-lg-3 row-cols-2 row-cols-md-3 row-cols-xl-4 justify-content-center">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->

			<!-- item_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
            {#each house_list as item}
            <ImageBlock {item}/>
            {/each}
		</div>
        <button class="prevent_btn" disabled id="nextbtn" on:click={post_member}>
            <div id="selectbtn_wrapper">
                <span>최소 5개 선택해 주세요.</span>
                <span></span>
                <span id="selected_num">Next!({selected_cnt}/5)</span>
            </div>
        </button>
	</div>
</section>





<style>
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
</style>