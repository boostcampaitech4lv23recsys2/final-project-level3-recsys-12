<script>
    import Like from './HomeElement/Like.svelte';
    import RecommendItem from './RecommendItem.svelte';

    let item_id = window.location.href.split('/').slice(-1)[0]
    let item = {}
    let price, rate, discount_price, category = []
	async function get_item() {
		// str로
        let url = import.meta.env.VITE_SERVER_URL + "/item"
		await fetch(url+"/"+parseInt(item_id), {
            headers: {
                Accept: "application / json",
            },
            method: "GET"
        }).then((response) => {
			response.json().then((json) => {
				item = json
                price = item.price ? Number(item.price.replace(/[^0-9]/g, "")) : 0
                rate = item.discount_rate ? Number(item.discount_rate.replace(/[^0-9]/g, "")/100) : 0
                discount_price = price*(1-rate) ? Number(price*(1-rate)).toLocaleString()+"원" : "미입점"
                price = price? price.toLocaleString()+"원" : ""
                category = item.category.split("|")
			})
		})
		.catch((error) => console.log(error))
	}
    get_item()

    let cluster_item = []
    async function get_cluster_item() {
        let url = import.meta.env.VITE_SERVER_URL + "/cluster/"+parseInt(item_id)
        await fetch(url, {
            headers: {
                Accept: "application / json",
            },
            method: "GET"
        }).then((response) => {
			response.json().then((json) => {
				cluster_item = json
			})
		})
		.catch((error) => console.log(error))
    }
    get_cluster_item()

    let series_item = []
    async function get_same_series_item() {
        let url = import.meta.env.VITE_SERVER_URL + "/series/"+parseInt(item_id)
        await fetch(url, {
            headers: {
                Accept: "application / json",
            },
            method: "GET"
        }).then((response) => {
			response.json().then((json) => {
				series_item = json
			})
		})
		.catch((error) => console.log(error))
    }
    get_same_series_item()

    let popular_item = []
    async function get_popular_item() {
        let url = import.meta.env.VITE_SERVER_URL + "/popular/"+parseInt(item_id)
        await fetch(url, {
            headers: {
                Accept: "application / json",
            },
            method: "GET"
        }).then((response) => {
			response.json().then((json) => {
				popular_item = json
			})
		})
		.catch((error) => console.log(error))
    }
    get_popular_item()

</script>

<section class="py-5">
    <div class="container px-4 px-lg-5 my-5">
        <div class="item-info">
            <div class="item-category">
                {#each category as cat, idx}
                <div class="each-category">{cat}</div>
                {#if idx != category.length - 1}
                <div class="each-category-sep">
                    &nbsp;>&nbsp;
                </div>
                {/if}
                {/each}

            </div>
            <div class="row gx-4 gx-lg-5 align-items-center">
                <div class="col-md-6"><Like item_id={item_id}/>
                    <img class="card-img-top mb-5 mb-md-0" src={item.image} alt="..." />
                </div>
                <div class="col-md-6">
                    <div class="item-seller mb-1">{item.seller}</div>
                    <a href="https://ohou.se/productions/{item.item_id}/selling">
                        <h1 class="item-title fw-bolder">{item.title}</h1>
                    </a>
                    <div class="fs-5 mb-5">
                        <span class="origin-price text-decoration-line-through">{price}</span>
                        <br>
                        <span class="discount-rate">{item.discount_rate}</span>
                        <span class="total-price">{discount_price}</span>
                    </div>
                    <div class="item-review">
                        <img class="star-icon" src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png" alt="...">
                    <span> 별점 {item.rating} <span style="color:gray;">|</span> {item.review}개 리뷰 </span>
                    </div>
                </div>
            </div>
        </div>
    
        <br><br><br>
        <!-- 같은 클러스터 상품 -->
        {#if cluster_item.length != 0}
        <div class="recom-item">
            이런 상품은 어때요?
        </div>
        <div class="item-view">
            <RecommendItem item_id_list={cluster_item} />
        </div>
        {/if}
        
        <!-- 같은 판매자가 판매하는 같은 카테고리 상품 -->
        {#if series_item.length != 0}
        <div class="recom-item">
            같은 시리즈 상품
        </div>

        <div class="item-view">
            <RecommendItem item_id_list={series_item} />
        </div>
        {/if}

        <!-- 같은 카테고리 내 인기 상품 -->
        {#if popular_item.length != 0}
        <div class="recom-item">
            {category[category.length - 1]} 인기 상품
        </div>
        <div class="item-view">
            <RecommendItem item_id_list={popular_item} />
        </div>
        {/if}
    </div>
</section>



<style>

    .item-info {
        margin-bottom: 10%;
    }
    .item-category {
        display: flex;
        vertical-align: center;
        margin-bottom: 1%;
    }
    .each-category-sep {
        font-weight: bold;
        color:gray;
        font-size: 1rem;
    }
    .each-category {
        font-size: 1.0rem;
        color: gray;
        font-weight: bold;
    }

    .card-img-top {
        border-radius: 5%;
    }


    .item-seller {
        font-size: 1.05rem;
        font-weight: 400;
    }
    a {
        text-decoration: none;
        color: black;
    }
    .item-title {
        font-size: 1.6rem;
    }
    .origin-price {
        color: gray;
        font-size: 1.0rem;
        font-weight: 500;
    }
    .discount-rate {
        font-size: 1.25rem;
        font-weight: bold;
        color: rgb(22, 176, 223);
    }
    .total-price {
        font-weight: bold;
        font-size: 1.4rem;
    }

    .item-review {
        font-size: 1.1rem;
        font-weight: 400;
    }

    .star-icon {
        width: 1.0rem;
        height: 1.0rem;
    }

    .recom-item {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 2%;
    }

    .recom-item {
        display: flex;
        justify-content: space-between;;
    }
    .item-view {
        margin-bottom: 7%;
    }

</style>