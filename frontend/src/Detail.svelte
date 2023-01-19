<hr>
<script>
	import { link } from 'svelte-spa-router'
    
    // let item = {}
	// async function get_item(item_id) {
	// 	// str로
	// 	await fetch("http://localhost:8000"+item_id).then((response) => {
	// 		response.json().then((json) => {
	// 			item = json
	// 		})
	// 	})
	// 	.catch((error) => console.log(error))
	// }
	// get_item()

    let item = {
        "item":57772,
        "category":"생활용품,수건·타월,세면타월",
        "rating":"별점 4.6점",
        "review":"(225)개 리뷰",
        "price":"61,300원",
        "title":"40수 200g 네이쳐 호텔수건 10매",
        "seller":"송월타올 ",
        "discount_rate":"10%",
        "image":"https://image.ohou.se/i/bucketplace-v2-development/uploads/openapi/1883336/1666575368641.jpg?gif=1&w=480&h=480&c=c",
        "available_product":null,
        "predict_price":null
    }
    let price = Number(item.price.replace(/[^0-9]/g, ""))
    let rate = Number(item.discount_rate.replace(/[^0-9]/g, "")/100)
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Roboto:400,500,700');
*{
    -webkit-box-sizing: border-box;
            box-sizing: border-box;
    margin: 0;
    padding: 0;
}
body{
    font-family: 'Roboto', sans-serif;
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

.badge {
    position: absolute;
    left: 0;
    top: 20px;
    text-transform: uppercase;
    font-size: 13px;
    font-weight: 700;
    background: red;
    color: #fff;
    padding: 3px 10px;
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

.product-links {
    text-align: right;
}

.product-links a {
    display: inline-block;
    margin-left: 5px;
    color: #e1e1e1;
    transition: 0.3s;
    font-size: 17px;
}

.product-links a:hover {
    color: #fbb72c;
}
#detail_wrapper{
    display:flex;
    flex-flow: row nowrap;
    justify-content: center;
}
.product_detail_sidebar{    
    width: 200px;
    position: relative;
    box-shadow: 0 2px 7px #dfdfdf;
    margin: 50px auto;
    background: #fafafa;
}
.test{
    background-color: gray;
}
</style>
<hr>
<div id="detail_wrapper">
    <div class="product-card">
        <div class="product-tumb">
            <img src={item.image} alt="">
        </div>
        <div class="product-details">
            <span class="product-catagory">{item.category}</span>
            <h4><a href="https://ohou.se/productions/{item.item}/selling">{item.title}</a></h4>
            <p>{item.rating}|{item.review}|{item.seller}</p>
            <div class="product-bottom-details">
                <div class="product-price"><small>{item.price}</small>{price*(1-rate)}원</div>
                <div class="product-links">
                    <a href=""><i class="fa fa-heart"></i></a>
                    <a href=""><i class="fa fa-shopping-cart"></i></a>
                </div>
            </div>
        </div>
    </div>
    <!-- <div class="product_detail_sidebar test">
        <dd>seller</dd>
    </div> -->
</div>