<script>

	import { link } from 'svelte-spa-router'    
	import { access_token, member_email, is_login, click_like_item_id } from './store'
	import Like from './HomeElement/Like.svelte';

	let item_list = []

	async function get_items() {
		let url
		if ($is_login) {
			if ($click_like_item_id != []) {
				url = "http://localhost:8000/insert-prefer/"+$member_email+"/"+$click_like_item_id
                await fetch(url).then((response) => {
                    response.json().then((json) => {
                        if (json == "failure") {
							console.log("이미 좋아요를 누른 아이템입니다.")
						} 
                    })
                })
				$click_like_item_id = ""
			}
			url = "http://localhost:8000/home/" + $member_email
		}
		else {
			url = "http://localhost:8000/home/"
		}
		await fetch(url).then((response) => {
			response.json().then((json) => {
				item_list = json
			})
		})
		.catch((error) => console.log(error))
	}
	get_items()

	function reg_exp_predict_price(predict_price) {
		if (predict_price == '') {
			return "정보없음"
		}
		const re = /.+(?=사)/
		return predict_price.match(re)[0].slice(3)
	}

</script>

<hr>

<!-- Section-->
<section class="py-3">
	<div class="container-md px-3 px-lg-3 mt-3">
		<div class="row gx-3 gx-lg-3 row-cols-2 row-cols-md-3 row-cols-xl-4 justify-content-center">
			<!-- 
				row-cols-n : 축소 화면에서 n개 보여줌
				row-cols-xl-n : 최대 화면에서 n개 보여줌
			-->

			<!-- item_list 반복문으로 탐색하며 이미지, 상품명, 가격 출력 -->
			{#each item_list as item}
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
										<h6 class="price">예상가 {reg_exp_predict_price(item.predict_price)}</h6>
										{:else}
										<h6 class="price">{item.price}</h6>
										{/if}
									</div>
								</div>
							</div>
						</a>
					</div>
			{/each}
		</div>
	</div>
</section>

<style>

	 /* a 태그의 파란색 글씨, 밑줄이 그어지는 것 제거 */
	.link-detail {
		text-decoration:none;
		color:black;
	}

	.seller {
		color: gray;
		font-size: 0.8rem;
		padding-bottom: 15px;
	}

	/* 가격 글자 색상 변경 */
	.price {
		color:gray;
		font-weight: bold;
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
		height: 80px;
	}

	.item-price {
		height: 10px;
	}

</style>