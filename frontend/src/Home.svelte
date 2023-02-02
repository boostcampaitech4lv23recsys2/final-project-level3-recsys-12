<script>

	import { link, push } from 'svelte-spa-router'    
	import { member_email, is_login, click_like_item_id } from './store'
	import Like from './HomeElement/Like.svelte';
    import TopButton from './GoTop.svelte'

	const heart_fill = "https://cdn-icons-png.flaticon.com/512/2589/2589054.png"
    const heart_not_fill = "https://cdn-icons-png.flaticon.com/512/2589/2589197.png"

	let member_name = $member_email.split('@')[0]

	let item_list = []
	let category_set = new Set()
	let category_list = []
	let price, rate, discount_price
	async function get_items() {
		
		let url = import.meta.env.VITE_SERVER_URL
		if ($is_login) {
			if ($click_like_item_id != "") {
				let _url = url + "/insert-prefer/"+$member_email+"/"+$click_like_item_id
                await fetch(_url, {
					headers: {
						Accept: "application/json",
					},
					method: "GET"
				}
				).then((response) => {
                    response.json().then((json) => {
                        if (json == "failure") {
							console.log("이미 좋아요를 누른 아이템입니다.")
						} 
                    })
                })
				$click_like_item_id = ""
			}
			url = url + "/home/" + $member_email
		}
		else {
			url = url + "/home"
		}
		await fetch(url, {
            headers: {
                Accept: "application / json",
            },
            method: "GET"
        }
		).then((response) => {
			response.json().then((json) => {
				item_list = json
				for (let item of item_list) {
					category_set.add(item.category.split('|')[0])
				}
				category_list = Array.from(category_set).sort();
			})
		})
		.catch((error) => console.log(error))
	}
	get_items()

	function get_discount_price(item) {
		price = item.price ? Number(item.price.replace(/[^0-9]/g, "")) : 0
		rate = item.discount_rate ? Number(item.discount_rate.replace(/[^0-9]/g, "")/100) : 0
		discount_price = price*(1-rate) ? Number(price*(1-rate)).toLocaleString()+"원" : "미입점"
		price = price? price : ""
		return discount_price
	}

	let new_item_list = []
	async function update_recom() {
		let url = import.meta.env.VITE_SERVER_URL+"/update-inference-result"
		let params = {
            "member_email" : $member_email
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
			if (json == "Already Update") {
              // 사용자가 좋아요 누른 상품이 없는 경우
              alert("더 많은 좋아요가 필요해요!")
              console.log(json)
              // inference 결과에 변경이 없는 경우
            }else {
              new_item_list = json.new_item
              item_list = json.inter
              console.log(new_item_list)
              console.log("업데이트 완료!")
              // window.location.reload()
              alert("추천이 완료되었습니다!")
				    }
			  })
     })
	}
	
	function nonlogin_recom() {
		let response = confirm("로그인이 필요한 서비스입니다. 로그인 하시겠습니까?")
		if (response) {
			push('/login-user')
		}
	}

	let first_img_url = heart_not_fill
	let second_img_url = heart_not_fill
	function change_heart_icon(num) {
		if (num == 1) {
			if (first_img_url == heart_fill) {
                first_img_url = heart_not_fill
			} else {
				first_img_url = heart_fill
			}
		}else {
			if (second_img_url == heart_fill) {
                second_img_url = heart_not_fill
			} else {
				second_img_url = heart_fill
			}
		}
	}

</script>

<!-- Section-->
<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
		{#if !$is_login}
		<div class="sticky-top refresh-button-wrapper">
			
			<button on:click="{nonlogin_recom}" class="refresh-button btn btn-outline-dark btn-lg" type="button">
				<img class="refresh-icon" src="https://cdn-icons-png.flaticon.com/512/179/179407.png" alt="...">
				<span class="refresh-button-text">개인 맞춤 추천 받기</span>
			</button>
		</div>

		<div class="description">
			<img class="refresh-icon" src="https://cdn-icons-png.flaticon.com/512/426/426833.png" alt="...">
			<span class="refresh-button-text">요즘 인기 있는 상품이에요</span>
		</div>
		{/if}
		{#if $is_login}
		<div class="sticky-top refresh-button-wrapper">
			
			<button on:click="{update_recom}" class="refresh-button btn btn-outline-dark btn-lg" type="button">
				<img class="refresh-icon" src="https://cdn-icons-png.flaticon.com/512/179/179407.png" alt="...">
				<span class="refresh-button-text">새로운 추천 받기</span>
			</button>
			

		</div>
		
		<div class="refresh-button-help">
			마음에 드는 상품에
			<img class="heart-icon" src={first_img_url} alt="..." on:mouseenter={() => {change_heart_icon(1)}} on:mouseleave={() => {change_heart_icon(1)}}>
			를 눌러주세요! <br>
			<img class="heart-icon" src={second_img_url} alt="..." on:mouseenter={() => {change_heart_icon(2)}} on:mouseleave={() => {change_heart_icon(2)}}>
			를 많이 고르면 더 좋은 추천 결과를 받아보실 수 있어요!
		</div>

		<div class="description">
			<img class="refresh-icon" src="https://cdn-icons-png.flaticon.com/512/456/456115.png" alt="...">
			<span class="refresh-button-text">{member_name}님이 좋아할 것 같은 상품들을 골라봤어요</span>
		</div>
		{/if}
		{#if new_item_list.length != 0}
		<div class="new-button" style="margin-top: 2rem;">
			<img class="new-icon" src="https://cdn-icons-png.flaticon.com/512/331/331953.png" alt="...">
			<span class="category-name refresh-button-text">새롭게 추천된 상품들이에요.</span>
		</div>
		<hr>
		<div class="row gx-3 gx-lg-3 row-cols-3 row-cols-md-4 row-cols-xl-5">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->
			
			{#each new_item_list as item}
			<div class="col mb-3">
				<Like item_id={item.item_id} />
				<a use:link href="/detail/{item.item_id}" class="link-detail">
					<div class="card h-100">
						<!-- Product image-->
						<img class="card-img-top" src={item.image} alt="..." />
						<!-- Product details-->
						<div class="card-body p-4">
							<!-- Product seller  -->
							<div class="seller">
								{item.seller}
							</div>
							<div class="item-name">
								<!-- Product name -->
								<h5 class="fw-bolder">{item.title}</h5>
							</div>
							<div class="text-center">
								<div class="item-price">
									<!-- Product price. 가격 정보가 없을 경우 미입점 처리 -->
									{#if item.price == ""}
									<h6 class="price">예상가 {item.predict_price}</h6>
									{:else}
									<h6 class="price">{get_discount_price(item)}</h6>
									{/if}
								</div>
							</div>
						</div>
					</a>
				</div>
			{/each}
		</div>
		{/if}
		{#each category_list as category}
		<div class="category-name">{category}</div>
		<hr>
		<div class="row gx-3 gx-lg-3 row-cols-3 row-cols-md-4 row-cols-xl-5">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->
			
			<!-- item_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
			
			{#each item_list as item}
			{#if item.category.split('|')[0] == category}
				<div class="col mb-3">
					<Like item_id={item.item_id} />
					<a use:link href="/detail/{item.item_id}" class="link-detail">
						<div class="card h-100">
							<!-- Product image-->
							<img class="card-img-top" src={item.image} alt="..." />
							<!-- Product details-->
							<div class="card-body p-4">
								<!-- Product seller  -->
								<div class="seller">
									{item.seller}
								</div>
								<div class="item-name">
									<!-- Product name -->
									<h5 class="fw-bolder">{item.title}</h5>
								</div>
								<div class="item-price">
									<div class="text-center">
										<!-- Product price. 가격 정보가 없을 경우 미입점 처리 -->
										{#if item.price == ""}
										<h6 class="price">예상가 {item.predict_price}</h6>
										{:else}
										<h6 class="price">{get_discount_price(item)}</h6>
										{/if}
									</div>
								</div>
							</div>
						</a>
					</div>
					{/if}
			{/each}
			
		</div>
		{/each}
	</div>
	<div class="go-top-button">
		<TopButton />
	</div>
</section>

<style>

	.refresh-button-wrapper {
		top: 120px;
		display: flex;
		justify-content: right;
		padding-bottom: 1rem;
	}
	.refresh-button-help {
		width: 100%;
		text-align: right;
		padding-bottom: 3rem;
	}

	.refresh-icon {
		width: 1.7rem;
		height: 1.7rem;
	}
	.refresh-button {
		color: black;
		background-color: white;
		display: flex;
		justify-content: center;
		font-weight: bold;
	}
	.new-icon {
		width: 1.7rem;
		height: 1.7rem;
	}
	.new-button {
		color: black;
		background-color: white;
		display: flex;
		justify-content: center;
		font-weight: bold;
	}
	.refresh-button:hover {
		background-color: white;
	}

	.refresh-button:hover .refresh-icon {
		animation-name: rotate_icon;
		animation-duration: 1.5s;
		animation-timing-function: ease-out;
		animation-iteration-count: infinite;
		transform-origin: 50% 50%;
	}

	@keyframes rotate_icon{
        100% {
			transform: rotate(360deg);
		}
    }

	.refresh-button-text {
		margin-left: 1rem;
		font-size: 1.1rem;
	}

	.heart-icon {
		width: 20px;
		height: 20px;
	}


	.description {
		font-size: 1.5rem;
		font-weight: bold;
		
		display: flex;
		justify-content: left;
		padding-top: 1rem;
		padding-bottom: 1.5rem;
	}

	.category-name {
		color: rgb(73, 72, 72);
		font-size: 1.2rem;
		font-weight: bold;
	}

	 /* a 태그의 파란색 글씨, 밑줄이 그어지는 것 제거 */
	.link-detail {
		text-decoration:none;
		color:black;
	}

	.seller {
		color: gray;
		font-size: 0.8rem;
		padding-bottom: 8%;
	}

	/* 가격 글자 색상 변경 */
	.price {
		color:gray;
		font-weight: bold;
		height: 10%;
		padding-bottom: 10%;
	}

	/* 마우스 오버 시 그림이 div보다 크게 scale되지 않도록 오버플로우 방지 */
	.card {
		overflow: hidden;
	}

	/* ======= 애니메이션 ========= */
	.link-detail .card-img-top {
		transition: transform .2s;
	}

	.link-detail .fw-bolder {
		transition: transform .2s;
	}
	.link-detail .seller {
		transition: transform .2s;
	}

	.link-detail:hover .card-img-top {
		transform: scale(1.05);
	}
	.link-detail:hover .fw-bolder {
		transform: scale(1.05);
	}
	.link-detail:hover .seller {
		transform: scale(1.03);
	}
	.link-detail:hover .price {
		color: #343a40;
	}
	/* ========================= */

	/* 상품명 글씨 크기 조정 */
	.fw-bolder {
  		font-size: 15px;
	}

	/* 상품명이 짧을 경우에도 price 위치 고정 */
	.item-name {
		height: 62%;
	}

	.go-top-button {
        position: fixed;
        right: 5%;
        bottom: 5%;
    }

</style>