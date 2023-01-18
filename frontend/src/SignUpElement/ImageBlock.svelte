<script>
    import { getContext } from "svelte";

	export let item;

	let img_opacity = 1;
	let selected_img = getContext("selected_img");
    let selected_cnt = getContext("selected_cnt");
	let img = document.querySelector("#house_img");
	
	const onHouseSelected = (e)=>{
		img_opacity = img_opacity == 0.5? 1:0.5;
		let nextbtn = document.querySelector("#nextbtn");
		// let selected_num = document.querySelector("#selected_num");
		if (selected_img.has(item.item_id)){
			selected_img.delete(item.item_id)
			e.target.style.borderWidth="0px";
		}else{
			selected_img.add(item.item_id)
			e.target.style.borderWidth="1px";
		};

		if (selected_img.size >= 5){
			nextbtn.classList.remove("prevent_btn");
			nextbtn.disabled = false
		}else{
			nextbtn.classList.add("prevent_btn");
			nextbtn.disabled = true
		};
		selected_cnt = Array.from(selected_img).length
		console.log(selected_cnt)
		// selected_num.textContent = "({0}/5)".format();
	}
</script>

<style>
	.house{
		margin:0;
		padding:0;
		width:20rem;
		/* height:10rem; */
		border-radius: 5px;
	}
	img{
		transition: opacity 0.2s;
		border: 0px solid black;
	}
</style>
<img src={item.image} on:click={onHouseSelected} alt="images" class="house" id="house_img" value={item.item_id} style:opacity={img_opacity}>
