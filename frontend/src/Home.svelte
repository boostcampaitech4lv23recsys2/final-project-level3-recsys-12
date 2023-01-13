<script>
	import { getContext } from 'svelte'
	import { link } from 'svelte-spa-router'

	let item_list

	fetch("http://localhost:8000/").then((response) => {
		console.log(response)
	})

	// 상위 컴포넌트(main.js)에서 전달한 props 가지고 옴.
	// const item_list = getContext('house_list')
</script>

<hr>

{#each item_list as item}
{item["funiture_name"]}
{/each}

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
					<a use:link href="/detail/{item.item}" class="link_detail">
						<div class="card h-100">
							<!-- Product image-->
							<img class="card-img-top" src={item.img_url} alt="..." />
							<!-- Product details-->
							<div class="card-body p-4">
								<div class="text-center">
									<div class="item-name">
										<!-- Product name-->
										<h5 class="fw-bolder">{item.name}</h5>
									</div>
									<div class="item-price">
										<!-- Product price. 미입점의 경우 뒤에 '원' 붙지 않도록 -->
										{#if item.price == "미입점"}
										<h6 class="price">{item.price}</h6>
										{:else}
										<h6 class="price">{item.price}원</h6>
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
	.link_detail {
		text-decoration:none;
		color:black;
	}

	/* 가격 글자 색상 변경 */
	.price {
		color:gray;
	}

	/* 마우스 오버 시 그림이 div보다 크게 scale되지 않도록 오버플로우 방지 */
	.card {
		overflow: hidden;
	}

	/* ======= 애니메이션 ========= */
	.link_detail .card-img-top {
		transition: transform .2s;
	}

	.link_detail .fw-bolder {
		transition: transform .2s;
	}

	.link_detail:hover .card-img-top {
		transform: scale(1.05);
	}
	.link_detail:hover .fw-bolder {
		transform: scale(1.05);
	}
	.link_detail:hover .price {
		color: #343a40;
	}
	/* ========================= */

	/* 상품명 글씨 크기 조정 */
	.fw-bolder {
  		font-size: 15px;
	}

	/* 상품명이 짧을 경우에도 price 위치 고정 */
	.item-name {
		height: 90px;
	}

	.item-price {
		height: 10px;
	}

</style>