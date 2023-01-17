<script>

	import { link } from 'svelte-spa-router'

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

    $: checked_house = [];
    
</script>

<hr>

<!-- Section-->
{checked_house}
<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
		<div class="row gx-3 gx-lg-3 row-cols-2 row-cols-md-3 row-cols-xl-4 justify-content-center">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->
			<!-- house_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
			{#each house_list as item}
            
                <div class="col mb-3">
                    <div class="card h-100">
                        <label>
                            <!-- Product image-->
                            <input class="check" type="checkbox" value="{item.item_id}" bind:group="{checked_house}">
                            <img class="each-image card-img-top" src={item.image} alt="..." />
                            <!-- Product details-->
                        </label>
                    </div>
                </div>
            {/each}
		</div>
	</div>
</section>

<style>

    

</style>