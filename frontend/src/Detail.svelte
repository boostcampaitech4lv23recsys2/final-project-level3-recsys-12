<script>
    import Like from './HomeElement/Like.svelte';

    let item_id = window.location.href.split('/').slice(-1)[0]
    let item = {}
    let price, rate, discount_price
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
                discount_price = price*(1-rate) ? parseInt(price*(1-rate))+"원" : "미입점"
                price = price? price : ""
			})
		})
		.catch((error) => console.log(error))
	}
    get_item()
</script>

<div id="detail_wrapper">
    <div class="product-card">
        <Like item_id={item_id} />
        <div class="product-tumb">
            <img src={item.image} alt="">
        </div>
        <div class="product-details">
            <span class="product-catagory">{item.category}</span>
            <h4><a href="https://ohou.se/productions/{item.item_id}/selling">{item.title}</a></h4>
            <p>{item.rating} | {item.review} | {item.seller}</p>
            <div class="product-bottom-details">
                <div class="product-price"><small>{price}</small>{discount_price}</div>
            </div>
        </div>
    </div>
</div>


<style>
@import url('https://fonts.googleapis.com/css?family=Roboto:400,500,700');
*{
    -webkit-box-sizing: border-box;
            box-sizing: border-box;
    margin: 0;
    padding: 0;
}

a{
    text-decoration: none;
}
.product-card {
    width: 380px;
    position: relative;
    box-shadow: 0 2px 7px #dfdfdf;
    margin: 50px auto;
    background: #fafafa;
}

.product-tumb {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 300px;
    padding: 50px;
    background: #f0f0f0;
}

.product-tumb img {
    max-width: 100%;
    max-height: 100%;
}

.product-details {
    padding: 30px;
}

.product-catagory {
    display: block;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    color: #ccc;
    margin-bottom: 18px;
}

.product-details h4 a {
    font-weight: 500;
    display: block;
    margin-bottom: 18px;
    text-transform: uppercase;
    color: #363636;
    text-decoration: none;
    transition: 0.3s;
}

.product-details h4 a:hover {
    color: #fbb72c;
}

.product-details p {
    font-size: 15px;
    line-height: 22px;
    margin-bottom: 18px;
    color: #999;
}

.product-bottom-details {
    overflow: hidden;
    border-top: 1px solid #eee;
    padding-top: 20px;
}

.product-bottom-details div {
    float: left;
    width: 50%;
}

.product-price {
    font-size: 18px;
    color: #fbb72c;
    font-weight: 600;
}

.product-price small {
    font-size: 80%;
    font-weight: 400;
    text-decoration: line-through;
    display: inline-block;
    margin-right: 5px;
}

#detail_wrapper{
    display:flex;
    flex-flow: row nowrap;
    justify-content: center;
}

</style>