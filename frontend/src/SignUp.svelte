<script>

	import { link } from 'svelte-spa-router'
    import ImageBlock from './SignUpElement/ImageBlock.svelte';
    import SignUpForm from './SignUpElement/SignUpForm.svelte';
    import {getContext, setContext} from "svelte"
	// let house_list = []

	// async function get_items() {
	// 	await fetch("http://localhost:8000/house_list").then((response) => {
	// 		response.json().then((json) => {
	// 			house_list = json
	// 		})
	// 	})
	// 	.catch((error) => console.log(error))
	// }
	// get_items()

    let house_list = [
        {"item_id": 133945, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/projects/cover_images/167016767527961408.jpg?w=960&h=640&c=c"},
        {"item_id": 114545, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167278499483240515.jpeg?w=960&h=640&c=c"},
        {"item_id": 134202, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167197656795637234.jpg?w=960&h=640&c=c"},
        {"item_id": 133943, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167082299072136488.jpg?w=960&h=640&c=c"},
        {"item_id": 133935, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167221022507849722.JPG?w=960&h=640&c=c"},
        {"item_id": 133924, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167056938562996617.jpg?w=960&h=640&c=c"},
        {"item_id": 134244, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167209746450457466.jpeg?w=960&h=640&c=c"},
        {"item_id": 133937, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/projects/cover_images/166844931137187017.jpg?w=960&h=640&c=c"},
        {"item_id": 133881, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167176043565523770.jpg?w=960&h=640&c=c"},
        {"item_id": 133786, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/projects/cover_images/167040020688209302.jpg?w=960&h=640&c=c"},
        {"item_id": 133710, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167022894551057322.jpg?w=960&h=640&c=c"},
        {"item_id": 133536, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167172468513982031.jpg?w=960&h=640&c=c"},
        {"item_id": 133732, "image": "https://image.ohou.se/i/bucketplace-v2-development/uploads/cards/projects/167091171836930326.heic?w=960&h=640&c=c"},
    ]

	let selected_img = new Set();
	setContext("selected_img",selected_img);

    let selected_cnt = 0;
    setContext("selected_cnt",selected_cnt);
</script>


<hr>
<section class="py-3">
<!-- <section> -->
    <SignUpForm/>
    <div class="container_title">
        <br>
        <h4 style="text-align: center; margin:0;">마음에 드는 이미지를 선택해 주세요.</h4>
        <br>
        <p style="text-align: center;">많이 선택할수록 개선된 결과가 표시됩니다.</p>
    </div>
	<div class="container_wrapper">
		<!-- 
			row-cols-n : 축소 화면에서 n개 보여줌
			row-cols-xl-n : 최대 화면에서 n개 보여줌
		-->
		<!-- house_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
		{#each house_list as item}
        <ImageBlock {item}/>
        {/each}
	</div>
    <button class="prevent_btn" disabled id="nextbtn">
        <div id="selectbtn_wrapper">
            <span>최소 5개 선택해 주세요.</span>
            <span></span>
            <span id="selected_num">Next!({selected_cnt}/5)</span>
        </div>
    </button>
</section>

<style>
#selectbtn_wrapper{
    display:flex;
    justify-content: space-evenly;
    opacity: 1;
}
</style>