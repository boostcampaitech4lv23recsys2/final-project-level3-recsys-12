<script>
    
    function goToTop(e) {
        e.preventDefault()
        window.scrollTo(0, 0)
    }

</script>

<div class="go-top-button">
    <button class="btn btn-secondary" on:click="{goToTop}">
        <img class="top-button" src="https://cdn-icons-png.flaticon.com/512/130/130906.png" alt="...">
        <br>Top
    </button>
</div>


<style>

.top-button {
        width: 20px;
        
    }

</style>