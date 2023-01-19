<script>
    import { getContext } from "svelte";

	export let item;

	let img_opacity = 1;
	let selected_img = getContext("selected_img");
    let selected_cnt = getContext("selected_cnt");
	let min_select = 1

	const onHouseSelected = (e)=>{
		img_opacity = img_opacity == 0.5? 1:0.5;
		let nextbtn = document.querySelector("#nextbtn");
		if (selected_img.has(item.house_id)){
			selected_img.delete(item.house_id)
			e.target.style.borderWidth="0px";
			// e.target.style.height="15rem";
		}else{
			selected_img.add(item.house_id)
			e.target.style.borderWidth="1px";
			// e.target.style.height="20rem";
		};

		if (selected_img.size >= min_select){
			nextbtn.classList.remove("prevent_btn");
			nextbtn.disabled = false
		}else{
			nextbtn.classList.add("prevent_btn");
			nextbtn.disabled = true
		};
		selected_cnt = Array.from(selected_img).length
		console.log(selected_cnt)
		console.log(selected_img)
		// selected_num.textContent = "({0}/5)".format();
	}
</script>

<style>
	.house{
		margin:0;
		padding:0;
		width:20rem;
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
</style>
<img 
	src={item.card_img_url} 
	on:click={onHouseSelected} 
	alt="images" 
	class="house" 
	id="house_img" 
	value={item.house_id} 
	style="opacity:{img_opacity}"
>
