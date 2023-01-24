<script>
    import { getContext } from "svelte";

	export let item;

	let img_opacity = 1;
	let selected_img = getContext("selected_img");
    let selected_cnt = getContext("selected_cnt");
	let min_select = 5
	const onHouseSelected = (e)=>{
		img_opacity = img_opacity == 0.5? 1:0.5;
		let nextbtn = document.querySelector(".nextbtn");
		if (selected_img.has(item.house_id)){
			selected_img.delete(item.house_id)
			e.target.style.borderWidth="0px";
		}else{
			selected_img.add(item.house_id)
			e.target.style.borderWidth="1px";
		}
		if (selected_img.size >= min_select){
			nextbtn.classList.remove("prevent_btn");
			nextbtn.disabled = false
		}else{
			nextbtn.classList.add("prevent_btn");
			nextbtn.disabled = true
		}
		selected_cnt = Array.from(selected_img).length
		document.querySelector("#selected_num").innerText = "Next!(" + parseInt(selected_cnt) + "/" + Math.max(5,selected_cnt) + ")"
	}
</script>

<style>
	.house{
		margin:0;
		padding:0;
		/* width:20rem; */
		height:15rem;
		border-radius: 5px;
	}
	img {
  		transition: all 0.2s linear;
	}
	img:hover{
    	transition: transform 0.2s linear;
    	transform-origin: 50% 50%;
    	transform: scale(1.05);
	  /* transform-origin: 100% 100%;
	  transform: scale(1.05); */
	  /* height: 20rem; */
	}
	img{
		transition: opacity 0.2s height 0.5s;
		border: 0px solid black;
	}
	#image_wrapper_button{
		border: 0;
		background-color: white;
		padding: 0;
		margin: 0;
	}
</style>
<button on:click={onHouseSelected} id="image_wrapper_button">
	<img 
		src={item.card_img_url} 
		alt="images" 
		class="house" 
		id="house_img" 
		value={item.house_id} 
		style="opacity:{img_opacity}"
	>
</button>
