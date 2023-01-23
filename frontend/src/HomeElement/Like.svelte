<script>
    import { push } from 'svelte-spa-router';


	import { access_token, member_email, is_login } from '../store'

    export let item_id
    let img_url
    let like_item

    if ($is_login) {
        // 이미 좋아요를 누른 경우 처리
        let url = "http://localhost:8000/prefer/"+$member_email+"/"+item_id
        fetch(url).then((response) => {
            response.json().then((json) => {
                if (json == true) {
                    img_url = "heart_fill.png"
                    like_item = true
                }
                else {
                    img_url = "heart_not_fill.png"
                    like_item = false
                }
			})
        })
    }
    else {
        img_url = "heart_not_fill.png"
    }

	async function change_like_icon() {
        // 이미지 좋아요 클릭한 순간에 아이콘 바뀌는 것 처리 필요.
		// if (img_url == "heart_fill.png") {
        //     img_url = "heart_not_fill.png"
        // } else {
        //     img_url = "heart_fill.png"
        // }
	}

    async function change_like_status(item_id) {
        let url
        if ($is_login) {
            if (like_item) {
                let params = {
                    member_email: $member_email,
                    item_id: JSON.stringify(item_id)
                }
                let options = {
                    method: "delete",
                    headers: {
                        "Content-Type": 'application/json'
                    },
                    body: JSON.stringify(params)
                }
                
                url = "http://localhost:8000/delete-prefer/"+$member_email+"/"+item_id
                fetch(url, options).then((response) => {
                    response.json().then((json) => {
                        like_item = false
                    })
                })
            }
            else {
                url = "http://localhost:8000/insert-prefer/"+$member_email+"/"+item_id
                fetch(url).then((response) => {
                    response.json().then((json) => {
                        like_item = true
                    })
                })
            }
            
            if (img_url == "heart_fill.png") {
                img_url = "heart_not_fill.png"
            } else {
                img_url = "heart_fill.png"
            }
        }
        else {
            let response = confirm("로그인이 필요합니다. 로그인 하시겠습니까?")
            if (response) {
                push('/login')
            }
            // 이 경우 비로그인 상태에서 누른 좋아요를 좋아요 리스트에 추가 필요
        }
        
}
</script>

<button on:click={() => change_like_status(item_id)}
    on:mouseenter={change_like_icon}
    on:mouseleave={change_like_icon}
    class="like-wrapper">
    <img class="like-icon" src={img_url} alt="...">
</button>


<style>

	.like-wrapper {
		z-index: 2;
		position: absolute;
		width: 30px;
		height: 30px;
		background-color: white;
		border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.5rem;
	}

    .like-icon {
        width: 20px;
        height: 20px;
    }

</style>