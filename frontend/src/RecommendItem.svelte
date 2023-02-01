<script>
	import TopButton from './GoTop.svelte'
    export let item_id_list

    let price, rate, discount_price
    function get_discount_price(item) {
		price = item.price ? Number(item.price.replace(/[^0-9]/g, "")) : 0
		rate = item.discount_rate ? Number(item.discount_rate.replace(/[^0-9]/g, "")/100) : 0
		discount_price = price*(1-rate) ? Number(price*(1-rate)).toLocaleString()+"원" : "미입점"
		price = price? price : ""
		return discount_price
	}

	let tmp = []
	let item_list = []
    async function get_item_info() {
		for (let item_id of item_id_list) {
			let url = import.meta.env.VITE_SERVER_URL + "/item/"+parseInt(item_id)
			await fetch(url, {
				headers: {
					Accept: "application / json",
				},
				method: "GET"
			}).then((response) => {
				response.json().then((json) => {
					tmp.push(json)
					item_list = tmp
				})
			})
			.catch((error) => console.log(error))
		}
    }
	get_item_info()

	const current_url = window.location.href.split('/').slice(0,-1).join('/')

	function on_item_click(e, item_id) {
		e.preventDefault()
		window.location.href = current_url + "/" + item_id
		window.location.reload()
	}

</script>

<div class="row gx-2 gx-lg-2 row-cols-3 row-cols-md-4 row-cols-xl-5">
    <!-- 
        row-cols-n : 축소 화면에서 n개 보여줌
        row-cols-xl-n : 최대 화면에서 n개 보여줌
    -->
    {#each item_list as item}
    <div class="col mb-3">
        <button on:click={(e)=>on_item_click(e, item.item_id)} class="link-detail">
			<div class="image h-100 move">
                <!-- Product image-->
                <img class="card-img-top move" src={item.image} alt="..." />
                <!-- Product details-->
                <div class="card-body move p-1">
                    <!-- Product seller  -->
                    <div class="seller move">
                        {item.seller}
                    </div>
                    <div class="item-name move">
                        <!-- Product name -->
                        <h5 class="fw-bolder move">{item.title}</h5>
                    </div>
                    <div class="text-center">
                        <div class="item-price move">
                            <!-- Product price. 가격 정보가 없을 경우 미입점 처리 -->
                            {#if item.price == ""}
                            <h6 class="price move">예상가 {item.predict_price}</h6>
                            {:else}
                            <h6 class="price move">{get_discount_price(item)}</h6>
                            {/if}
                        </div>
                    </div>
                </div>
			</button>
        </div>
    {/each}
</div>
<div class="go-top-button">
	<TopButton />
</div>

<style>

	.link-detail {
		position: relative;
		border: 0;
		background-color: white;
		
	}
    .card-img-top {
        border-radius: 10%;
    }

	.seller {
		color: gray;
		font-size: 0.8rem;
		padding-bottom: 3%;
	}

	/* 가격 글자 색상 변경 */
	.price {
		color:gray;
		font-weight: bold;
		height: 10%;
		padding-bottom: 10%;
	}

	/* 마우스 오버 시 그림이 div보다 크게 scale되지 않도록 오버플로우 방지 */
	.image {
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
		height: 50%;
	}

	.go-top-button {
        position: fixed;
        right: 5%;
        bottom: 5%;
    }

</style>